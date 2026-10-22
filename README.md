# Multimodal Investment Scam Detector

Detects investment scams (Ponzi/"paisa double" schemes, fake trading signals, crypto
pig-butchering, withdrawal-fee traps) by recognising patterns across **three modalities**
and fusing them.

```
 text  ──► scam-pattern lexicon + TF-IDF ──► LogisticRegression ─┐
 image ──► OpenCV visual features (+OCR) ──► RandomForest ───────┼─► meta-learner ──► scam probability
 meta  ──► URL / sender / payment features ─► GradientBoosting ──┘   (trained with      + explanation
                                                                      modality dropout)
```

| Modality | Patterns recognised |
|---|---|
| Text | guaranteed / risk-free returns, unrealistic % (e.g. "20% daily"), urgency, move to Telegram/WhatsApp, USDT/crypto/forex, registration or withdrawal fees, fake SEBI/RBI authority, romance/"wrong number" lures |
| Image | green "profit" dominance, rising-chart trend, text-heavy screenshots, near-duplicates of known scam images (perceptual hash), optional OCR text |
| Metadata | link shorteners, suspicious TLDs, raw-IP links, lure words in domain, new accounts, follower ratio, channel, crypto/gift-card payment |

Two fusion strategies:
- `stack` (default): stacking ensemble, works when any modality is missing, gives a per-pattern explanation.
- `neural`: PyTorch attention-fusion network, reports how much attention each modality received.

## Quick start
```bash
pip install -r requirements.txt
python generate_data.py --n 1200          # synthetic demo data -> data/dataset.csv
python train.py                           # stacking model
python train.py --model neural            # attention-fusion model
python predict.py --text "Earn 30% daily, guaranteed! Join Telegram" \
    --url http://bit.ly/x --channel telegram --account-age 4 --payment crypto
```

## Training on public real-world data
```bash
python download_data.py        # ~17 MB into data/raw/ (data files only)
python prepare_real_data.py    # merge -> data/real_dataset.csv
python train.py --data data/real_dataset.csv --out models/scam_detector_real.joblib
```

| Source | Modality | Rows used | Notes |
|---|---|---|---|
| [Shade63/scam-classification-multiclass](https://huggingface.co/datasets/Shade63/scam-classification-multiclass) | text | 14,000 | Indian SMS/email incl. investment & crypto scams; appears template-generated |
| [bolewara/hinglish-scam-text-dataset](https://huggingface.co/bolewara/hinglish-scam-text-dataset) | text | ~3,550 | UCI SMS Spam Collection + hand-written Hinglish (CC-BY-4.0) |
| [CryptoScamDB blacklist](https://github.com/CryptoScamDB/blacklist) | url | 3,000 | crypto scam domains, "Trust-Trading"/"Investment" first |
| [OpenPhish feed](https://openphish.com/) | url | ~260 | live phishing URLs |
| [Tranco top sites](https://tranco-list.eu/) | url | ~3,260 | legitimate domains |
| your screenshots in `data/raw/images/{scam,legit}/` | image | — | no public investment-scam image set exists |

Near-duplicate messages are grouped by template and kept on one side of the train/test split.

Results (25% held-out groups):

| Metric | Value |
|---|---|
| ROC-AUC (all) | 0.984 |
| Accuracy (all) | 0.930 |
| Recall on investment-related scams | 0.691 |
| False-positive rate on legitimate messages | 0.036 |
| Text-expert AUC | 0.999 |
| URL/metadata-expert AUC | 0.685 |

Caveats: the Shade63 texts are template-based, so the near-perfect text score is optimistic; the
URL-only rows (domain without message) are the hard part and pull investment recall down.

## Using your own data
Put rows in `data/raw/my_samples.csv` (merged automatically) or train on any CSV with these columns
(any may be empty except `label`):

`text, image_path, url, channel, account_age_days, followers, following, verified, sender_in_contacts, payment_method, label`

- `channel`: telegram / whatsapp / sms / email / instagram / facebook / twitter / other
- `payment_method`: none / bank / upi / card / crypto / gift_card
- `label`: 1 = scam, 0 = legitimate

Possible sources: reported scam messages/screenshots (with consent), public SMS-spam and
phishing-URL datasets, and regulator scam advisories, paired with genuine broker/bank/news messages.

## Notes
- Results on the synthetic data only show that the pipeline works. Measure on real, held-out data before relying on it.
- OCR (`--ocr`) needs `pip install pytesseract` plus the Tesseract binary.
- Upgrades: swap TF-IDF for a multilingual transformer (e.g. IndicBERT/XLM-R) and the image features
  for CLIP embeddings; the fusion layer stays the same.
