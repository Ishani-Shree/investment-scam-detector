"""Merge the downloaded public datasets (and your own samples) into data/real_dataset.csv.

Sources (put under data/raw/, see README):
  shade63/sentinel_dataset_multiclass.csv  Indian SMS/email scams, incl. investment & crypto  (text)
  hinglish/data.json                       UCI SMS spam + hand-written Hinglish fraud          (text)
  cryptoscamdb/urls.yaml                   crypto scam / phishing domains                      (url, label 1)
  openphish/feed.txt                       live phishing URLs                                  (url, label 1)
  tranco/top-1m.csv.zip                    popular legitimate domains                          (url, label 0)
  images/scam/*, images/legit/*            screenshots you collected                           (image)
  my_samples.csv                           your own rows in the project CSV format             (any)

Every row gets a `group` id: near-identical (template) messages share a group so
train/test splits never leak copies of the same template.
"""
import argparse
import glob
import json
import os
import random
import re
import zipfile
from urllib.parse import urlparse

import pandas as pd
import yaml

RAW = "data/raw"
INVEST_RE = re.compile(r"\binvest(?:ment|ing|or|ors|ed)?s?\b|\btrad(?:ing|er|ers)\b|\bforex\b|\busdt\b|"
                       r"\bstocks?\b|share market|\breturns\b|\bprofits?\b|double your|\bsignals?\b|"
                       r"\bportfolio\b|\bipo\b|mutual funds?|\bsip\b|\bdemat\b|\bnifty\b|\bsensex\b|"
                       r"crypto (?:fund|trading|mining|arbitrage|signal)|\bmining pool\b|\bpresale\b", re.I)
NON_INVEST_CATEGORIES = {"government_impersonation", "bank_phishing"}
URL_RE = re.compile(r"((?:https?://)?(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:in|com|net|org|co|xyz|top|online|"
                    r"site|info|me|ly|io|app|vip|live|club|shop|biz|cc|uk)\b(?:/[^\s,]*)?)", re.I)
COLUMNS = ["text", "image_path", "url", "channel", "account_age_days", "followers", "following", "verified",
           "sender_in_contacts", "payment_method", "label", "category", "is_investment", "source", "group"]


def template_key(text):
    t = re.sub(r"[\d,.]+", "#", str(text).lower())
    t = URL_RE.sub("URL", t)
    return " ".join(t.split()[:8])


def host_of(url):
    u = url.strip()
    host = urlparse(u if "://" in u else "http://" + u).hostname or ""
    return host[4:] if host.startswith("www.") else host


def text_row(text, label, category, source):
    m = URL_RE.search(text)
    return dict(text=text, url=m.group(1) if m else "", channel="sms", label=int(label), category=category,
                is_investment=int(category not in NON_INVEST_CATEGORIES and bool(INVEST_RE.search(text))),
                source=source)


def load_shade63():
    p = f"{RAW}/shade63/sentinel_dataset_multiclass.csv"
    if not os.path.exists(p):
        return []
    df = pd.read_csv(p)
    return [text_row(t, l != 0, c, "shade63") for t, l, c in zip(df.text, df.label, df.category_name)]


def load_hinglish():
    p = f"{RAW}/hinglish/data.json"
    if not os.path.exists(p):
        return []
    rows = json.load(open(p, encoding="utf-8"))["rows"]
    return [text_row(r["text"], r["label"], "sms_spam" if r["label"] else "legit", "hinglish") for r in rows]


def url_row(host, label, category, source):
    # Bare host for both classes, so scheme/path differences between sources can't leak the label.
    return dict(text="", url=host, channel="other", label=label, category=category,
                is_investment=int(category in ("Trust-Trading", "Investment")), source=source)


def load_scam_urls(n, rng):
    rows, seen = [], set()
    p = f"{RAW}/cryptoscamdb/urls.yaml"
    if os.path.exists(p):
        entries = yaml.safe_load(open(p, encoding="utf-8")) or []
        # investment-style scams first, then fill with other crypto phishing
        entries.sort(key=lambda e: (e.get("subcategory") not in ("Trust-Trading", "Investment"), rng.random()))
        for e in entries:
            h = host_of(str(e.get("url") or e.get("name") or ""))
            if h and h not in seen and len(rows) < n:
                seen.add(h)
                rows.append(url_row(h, 1, str(e.get("subcategory") or e.get("category")), "cryptoscamdb"))
    p = f"{RAW}/openphish/feed.txt"
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            h = host_of(line)
            if h and h not in seen:
                seen.add(h)
                rows.append(url_row(h, 1, "phishing", "openphish"))
    return rows, seen


def load_legit_urls(n, exclude, rng):
    p = f"{RAW}/tranco/top-1m.csv.zip"
    if not os.path.exists(p):
        return []
    with zipfile.ZipFile(p) as z:
        lines = z.read(z.namelist()[0]).decode().splitlines()[:200_000]
    hosts = [l.split(",", 1)[1].strip() for l in lines if "," in l]
    hosts = [h for h in hosts if h not in exclude]
    return [url_row(h, 0, "legit", "tranco") for h in rng.sample(hosts, min(n, len(hosts)))]


def load_images():
    rows = []
    for label, folder in ((1, "scam"), (0, "legit")):
        for p in sorted(glob.glob(f"{RAW}/images/{folder}/*")):
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                rows.append(dict(text="", image_path=p.replace("\\", "/"), label=label,
                                 category=folder, source="my_images"))
    return rows


def load_my_samples():
    p = f"{RAW}/my_samples.csv"
    if not os.path.exists(p):
        return []
    df = pd.read_csv(p, keep_default_na=False)
    df["source"] = "my_samples"
    return df.to_dict("records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-urls", type=int, default=3000, help="URL-only rows per class")
    ap.add_argument("--out", default="data/real_dataset.csv")
    a = ap.parse_args()
    rng = random.Random(0)

    rows = load_shade63() + load_hinglish()
    scam_urls, scam_hosts = load_scam_urls(a.n_urls, rng)
    rows += scam_urls + load_legit_urls(len(scam_urls), scam_hosts, rng) + load_images() + load_my_samples()

    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    df["account_age_days"] = df["account_age_days"].fillna(-1)
    df["payment_method"] = df["payment_method"].fillna("none")
    df["channel"] = df["channel"].fillna("other")
    df = df.fillna({"followers": 0, "following": 0, "verified": 0, "sender_in_contacts": 0,
                    "is_investment": 0}).fillna("")

    has_text = df.text.str.strip() != ""
    df = df[~(has_text & df.text.str.lower().str.strip().duplicated())]  # exact text duplicates
    has_text = df.text.str.strip() != ""
    df["group"] = df.text.map(template_key).where(has_text, df.url.where(df.url != "", df.image_path))
    df["group"] = pd.factorize(df["group"])[0]
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    df.to_csv(a.out, index=False, encoding="utf-8")

    print(f"wrote {len(df)} rows -> {a.out}  ({df.label.mean():.0%} scam, {df.group.nunique()} groups)")
    print(pd.crosstab(df.source, df.label.map({0: "legit", 1: "scam"})))
    print("investment-related scam rows:", int(((df.label == 1) & (df.is_investment == 1)).sum()))
    print("rows with image:", int((df.image_path != "").sum()))


if __name__ == "__main__":
    main()
