"""Metadata modality: URL, sender profile, channel and payment-method signals."""
import math
import re
from urllib.parse import urlparse

import numpy as np

SUSPICIOUS_TLDS = {"xyz", "top", "club", "online", "site", "icu", "buzz", "vip", "live", "cc",
                   "shop", "win", "loan", "click", "work", "info", "biz"}
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "rb.gy", "shorturl.at"}
LURE_WORDS = ["invest", "profit", "trade", "crypto", "earn", "wealth", "fx", "bonus", "secure",
              "official", "verify", "wallet", "sebi", "rbi"]
CHANNELS = ["telegram", "whatsapp", "sms", "email", "instagram", "facebook", "twitter", "other"]
PAYMENTS = ["none", "bank", "upi", "card", "crypto", "gift_card"]


def url_features(url):
    url = (url or "").strip()
    if not url:
        return [0.0] * 11
    p = urlparse(url if "://" in url else "http://" + url)
    host = (p.hostname or "").lower()
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    return [
        1.0,
        math.log1p(len(url)),
        host.count("."),
        float(bool(re.fullmatch(r"[\d.]+", host))),
        float(host in SHORTENERS),
        float(tld in SUSPICIOUS_TLDS),
        float(p.scheme == "https"),
        float(bool(re.search(r"\d", host))),
        host.count("-"),
        sum(w in host for w in LURE_WORDS),
        float("@" in url or "//" in p.path),
    ]


def meta_vector(row):
    def num(key, default=0.0):
        try:
            v = float(row.get(key, default))
            return default if math.isnan(v) else v
        except (TypeError, ValueError):
            return default

    age = num("account_age_days", -1)
    followers = num("followers")
    following = num("following")
    channel = str(row.get("channel") or "other").lower()
    payment = str(row.get("payment_method") or "none").lower()
    feats = url_features(row.get("url"))
    feats += [
        math.log1p(max(age, 0)), float(age < 0), float(0 <= age < 30),
        math.log1p(followers), math.log1p(following),
        math.log1p(following / (followers + 1)),
        num("verified"), num("sender_in_contacts"),
    ]
    feats += [float(channel == c) for c in CHANNELS]
    feats += [float(payment == p) for p in PAYMENTS]
    return np.array(feats, dtype=float)


def meta_flags(row):
    """Human-readable red flags for explanations."""
    flags = []
    f = url_features(row.get("url"))
    if f[0]:
        if f[4]: flags.append("URL uses a link shortener")
        if f[5]: flags.append("URL has a suspicious TLD")
        if f[3]: flags.append("URL is a raw IP address")
        if f[9]: flags.append("domain contains lure words (invest/profit/crypto...)")
        if not f[6]: flags.append("link is not HTTPS")
    try:
        age = float(row.get("account_age_days", -1))
        if 0 <= age < 30:
            flags.append(f"sender account is only {int(age)} days old")
    except (TypeError, ValueError):
        pass
    pay = str(row.get("payment_method") or "").lower()
    if pay in ("crypto", "gift_card"):
        flags.append(f"asks for payment via {pay.replace('_', ' ')}")
    return flags
