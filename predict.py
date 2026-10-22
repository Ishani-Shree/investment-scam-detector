"""Score a single message (any combination of modalities).

Example:
  python predict.py --text "Earn 20% daily, guaranteed! Join Telegram" \
      --image screenshot.png --url http://bit.ly/x --channel telegram \
      --account-age 5 --payment crypto
"""
import argparse
import json

import pandas as pd

from scamdetect.fusion import MultimodalScamDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/scam_detector_stack.joblib")
    ap.add_argument("--text", default="")
    ap.add_argument("--image", default="")
    ap.add_argument("--url", default="")
    ap.add_argument("--channel", default="other")
    ap.add_argument("--account-age", type=float, default=-1)
    ap.add_argument("--followers", type=float, default=0)
    ap.add_argument("--following", type=float, default=0)
    ap.add_argument("--verified", type=int, default=0)
    ap.add_argument("--in-contacts", type=int, default=0)
    ap.add_argument("--payment", default="none",
                    choices=["none", "bank", "upi", "card", "crypto", "gift_card"])
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()

    model = MultimodalScamDetector.load(a.model)
    row = pd.DataFrame([dict(text=a.text, image_path=a.image, url=a.url, channel=a.channel,
                             account_age_days=a.account_age, followers=a.followers, following=a.following,
                             verified=a.verified, sender_in_contacts=a.in_contacts, payment_method=a.payment)])
    if hasattr(model, "net_"):  # neural fusion model
        p, att = model.predict_with_attention(row)
        print(json.dumps({"scam_probability": round(float(p[0]), 4), "modality_attention": att[0]}, indent=2))
    else:
        print(json.dumps(model.explain(row, a.threshold)[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
