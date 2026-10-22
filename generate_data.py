"""Generate a synthetic multimodal dataset (text + image + metadata) for development.

Replace with real data later: just keep the same CSV columns (see README).
Usage:  python generate_data.py --n 1200
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

R = random.Random(0)
NP = np.random.default_rng(0)

SCAM_TEXT = [
    "Earn {pct}% daily profit with our {asset} trading bot! 100% guaranteed returns, zero loss. Join our Telegram {tg} now",
    "Dear, my uncle works at a big fund and gave me an insider tip. Invest ₹{amt} today and double your money in {d} days. Trust me",
    "Limited slots! VIP signals group by expert mentor. {pct}% weekly returns assured. DM me on WhatsApp {ph}",
    "Congratulations! Your {asset} wallet has ₹{amt} profit. Pay withdrawal tax of ₹{fee} to unlock your funds. Hurry, expires today",
    "SEBI registered advisor 🚀🚀 My students earned ₹{amt} last week. Registration fee only ₹{fee}. Act now!!!",
    "New coin presale, {x}x returns guaranteed 💰💰 Send USDT to wallet address {wal} before it's too late",
    "Paisa double scheme! Invest ₹{amt}, get ₹{amt2} in {d} days. Risk free. Pay via UPI {upi}",
    "Hi, sorry wrong number. By the way I trade {asset} and make {pct}% per week, I will teach you. Add me on Telegram",
    "Your account frozen due to pending verification. Deposit first ₹{fee} activation fee to continue earning {pct}% daily",
    "Join our crypto mining pool, earn ₹{amt} every day without work. Only today bonus! {url}",
    "Hello, our team gives stock tips with good accuracy. Join group for free tips {url}",  # subtle
    "Investment plan: put ₹{amt} and receive monthly payout of ₹{amt2}. Contact {ph}",     # subtle
]
LEGIT_TEXT = [
    "Nifty closed {pct2}% lower today as IT stocks dragged; analysts expect volatility ahead of the RBI policy.",
    "Reminder: your SIP of ₹{amt} in the index fund will be debited on the 5th. Past returns do not guarantee future results.",
    "Quarterly results: the company reported revenue growth of {pct2}% YoY, margins were flat. Full report on our website.",
    "Webinar this Saturday: basics of mutual funds, asset allocation and risk. Free registration on {url}",
    "Your demat account statement for the month is ready. Log in to the app to view holdings.",
    "Bitcoin fell {pct2}% this week amid regulatory concerns. Crypto assets are highly volatile, invest only what you can afford to lose.",
    "Our research note on {asset}: we maintain a neutral rating with a target price revision. Read the disclosures.",
    "FD rates revised: {pct2}% p.a. for 1-year deposits, effective from Monday.",
    "Tax saving tip: ELSS funds have a 3-year lock-in. Consult a registered advisor before investing.",
    "Thanks for joining our Telegram channel for market news updates. We never ask for money or OTPs.",  # hard
    "Market recap: Sensex up {pct2}%, banking led the gains. Returns are subject to market risk.",
    "Dear customer, the bank never asks for your PIN. Report suspicious calls to the cyber helpline 1930.",  # hard
]
ASSETS = ["bitcoin", "forex", "gold", "crypto", "stock", "options", "BTC", "Nifty"]
SCAM_URLS = ["http://bit.ly/{r}", "http://profit-invest{n}.xyz/join", "https://crypto-earn.top/{r}",
             "http://secure-wealth-fx.online", "http://45.{n}.12.9/login", "https://sebi-official-verify.site"]
LEGIT_URLS = ["https://www.nseindia.com", "https://zerodha.com/varsity", "https://www.moneycontrol.com/news",
              "https://groww.in/mutual-funds", "https://www.rbi.org.in", "https://www.sebi.gov.in"]


def fill(t):
    return t.format(pct=R.choice([5, 8, 10, 15, 20, 30, 50]), pct2=round(R.uniform(0.2, 4), 1),
                    asset=R.choice(ASSETS), amt=f"{R.choice([5, 10, 25, 50, 100])},000",
                    amt2=f"{R.choice([2, 3, 5])},00,000", fee=R.choice([999, 1500, 4999, 9999]),
                    d=R.choice([7, 10, 15, 30]), x=R.choice([10, 20, 50, 100]), tg="t.me/" + rid(),
                    ph=f"+91 9{R.randint(100000000, 999999999)}", upi=rid() + "@ybl",
                    wal="0x" + "".join(R.choice("abcdef0123456789") for _ in range(12)),
                    url=R.choice(SCAM_URLS + LEGIT_URLS).format(r=rid(), n=R.randint(1, 99)))


def rid():
    return "".join(R.choice("abcdefghijkmnpqrstuvwxyz23456789") for _ in range(6))


def font(sz):
    return ImageFont.load_default(size=sz)


def line_chart(d, box, trend, color_up=(0, 200, 80), color_dn=(220, 40, 40)):
    x0, y0, x1, y1 = box
    n = 40
    steps = NP.normal(trend, 1.0, n)
    ys = np.cumsum(steps)
    ys = (ys - ys.min()) / (np.ptp(ys) + 1e-6)
    pts = [(x0 + i * (x1 - x0) / (n - 1), y1 - v * (y1 - y0)) for i, v in enumerate(ys)]
    for a, b in zip(pts, pts[1:]):
        d.line([a, b], fill=color_up if b[1] <= a[1] else color_dn, width=4)


def scam_image(path):
    kind = R.choice(["profit", "banner", "withdraw"])
    bg = R.choice([(10, 14, 30), (0, 0, 0), (20, 30, 20)]) if kind != "banner" else (R.randint(150, 230), 140, 0)
    im = Image.new("RGB", (400, 400), bg)
    d = ImageDraw.Draw(im)
    if kind == "profit":
        d.text((30, 30), f"+{R.randint(150, 900)}%", fill=(0, 230, 90), font=font(64))
        d.text((30, 110), f"Profit: ₹{R.randint(1, 99)},{R.randint(100, 999)},000", fill=(0, 230, 90), font=font(26))
        line_chart(d, (30, 180, 370, 370), trend=0.9)
    elif kind == "banner":
        d.text((20, 40), "DOUBLE YOUR", fill=(255, 255, 255), font=font(44))
        d.text((20, 95), "MONEY!!", fill=(0, 120, 0), font=font(60))
        d.text((20, 200), "JOIN TELEGRAM VIP", fill=(255, 255, 255), font=font(30))
        d.text((20, 260), "100% GUARANTEED", fill=(0, 90, 0), font=font(32))
    else:
        d.rectangle((40, 60, 360, 340), fill=(25, 35, 50))
        d.ellipse((160, 90, 240, 170), fill=(0, 200, 80))
        d.text((70, 190), "Withdrawal Successful", fill=(0, 220, 90), font=font(26))
        d.text((110, 240), f"₹{R.randint(1, 9)},{R.randint(10, 99)},000", fill=(255, 255, 255), font=font(40))
    im.save(path)


def legit_image(path):
    kind = R.choice(["chart", "article", "photo"])
    if kind == "chart":
        im = Image.new("RGB", (400, 400), (255, 255, 255))
        d = ImageDraw.Draw(im)
        for y in range(60, 380, 40):
            d.line([(20, y), (380, y)], fill=(225, 225, 225))
        d.text((20, 15), R.choice(["NIFTY 50", "SENSEX", "BANKNIFTY", "Portfolio"]), fill=(40, 40, 40), font=font(22))
        line_chart(d, (20, 60, 380, 380), trend=R.uniform(-0.3, 0.3))
    elif kind == "article":
        im = Image.new("RGB", (400, 400), (250, 250, 248))
        d = ImageDraw.Draw(im)
        d.text((20, 20), "Market News", fill=(20, 20, 60), font=font(30))
        for y in range(80, 380, 22):
            d.line([(20, y), (R.randint(250, 380), y)], fill=(130, 130, 130), width=6)
    else:
        a = NP.integers(0, 255, 3)
        b = NP.integers(0, 255, 3)
        g = np.linspace(0, 1, 400)[:, None, None]
        arr = (a * (1 - g) + b * g).repeat(400, 1).astype(np.uint8)
        im = Image.fromarray(arr)
        d = ImageDraw.Draw(im)
        for _ in range(6):
            x, y, r = R.randint(0, 400), R.randint(0, 400), R.randint(20, 80)
            d.ellipse((x - r, y - r, x + r, y + r), fill=tuple(int(v) for v in NP.integers(0, 255, 3)))
    im.save(path)


def meta(label):
    scam = label == 1
    if R.random() < 0.15:  # overlap: some scammers look legit and vice versa
        scam = not scam
    return dict(
        url=(R.choice(SCAM_URLS) if scam else R.choice(LEGIT_URLS)).format(r=rid(), n=R.randint(1, 99))
        if R.random() < 0.7 else "",
        channel=R.choice(["telegram", "whatsapp", "instagram", "sms"] if scam else ["email", "sms", "twitter", "facebook"]),
        account_age_days=int(NP.exponential(40 if scam else 900)),
        followers=int(NP.exponential(150 if scam else 5000)),
        following=int(NP.exponential(1500 if scam else 300)),
        verified=int(R.random() < (0.05 if scam else 0.5)),
        sender_in_contacts=int(R.random() < (0.1 if scam else 0.4)),
        payment_method=R.choice(["crypto", "upi", "gift_card", "upi", "bank"] if scam else ["none", "bank", "none", "card"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    img_dir = os.path.join(a.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    for i in range(a.n):
        label = int(R.random() < 0.45)
        # 15% "camouflaged" text: scams written like legit notices and vice versa
        style = label if R.random() > 0.15 else 1 - label
        text = fill(R.choice(SCAM_TEXT if style else LEGIT_TEXT)) if R.random() > 0.05 else ""
        img = ""
        if R.random() < 0.7:
            img = os.path.join(img_dir, f"{i:05d}.png")
            style_scam = label if R.random() > 0.1 else 1 - label
            (scam_image if style_scam else legit_image)(img)
        rows.append(dict(id=i, text=text, image_path=img, **meta(label), label=label))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "dataset.csv"), index=False, encoding="utf-8")
    print(f"wrote {len(df)} rows ({df.label.mean():.0%} scam) to {a.out}/dataset.csv")


if __name__ == "__main__":
    main()
