"""Web demo: paste a message, add a link / screenshot / sender details, get an explained verdict.

Usage:  python app.py   then open http://127.0.0.1:5000
"""
import argparse
import os
import tempfile

import scamdetect  # noqa: F401  (sets LOKY_MAX_CPU_COUNT before sklearn loads)
import pandas as pd
from flask import Flask, jsonify, render_template, request

from scamdetect.fusion import MODALITIES, MultimodalScamDetector

MODEL_CANDIDATES = ["models/scam_detector_real.joblib", "models/scam_detector_stack.joblib"]
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB uploads
MODEL = None
MODEL_PATH = None

EXAMPLES = {
    "scam": dict(text="Join our Zerodha VIP signals group 🚀 30% monthly returns guaranteed, zero loss. "
                      "Limited slots! Pay registration fee via UPI and DM me on Telegram.",
                 url="http://zerodha-vip-signals.in/join", channel="telegram", account_age_days=6,
                 payment_method="upi"),
    "legit": dict(text="Your SIP of Rs 5,000 in the index fund will be debited on the 5th. "
                       "Mutual fund investments are subject to market risks.",
                  url="https://groww.in/mutual-funds", channel="sms", account_age_days=1500,
                  payment_method="bank"),
}


def load_model(path=None):
    global MODEL, MODEL_PATH
    for p in [path] if path else MODEL_CANDIDATES:
        if p and os.path.exists(p):
            MODEL, MODEL_PATH = MultimodalScamDetector.load(p), p
            return
    raise SystemExit("No trained model found. Run: python train.py --data data/real_dataset.csv "
                     "--out models/scam_detector_real.joblib")


def _num(v, default):
    try:
        return float(v) if v is not None and str(v).strip() != "" else default
    except (TypeError, ValueError):
        return default


@app.get("/")
def index():
    return render_template("index.html", examples=EXAMPLES, model_path=MODEL_PATH,
                           image_enabled=MODEL.image_model_ is not None)


@app.post("/analyze")
def analyze():
    f = request.form
    tmp_path = ""
    upload = request.files.get("image")
    if upload and upload.filename:
        ext = os.path.splitext(upload.filename)[1].lower()
        if ext not in IMAGE_EXT:
            return jsonify(error=f"Unsupported image type {ext}"), 400
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        upload.save(tmp_path)
    try:
        row = dict(text=f.get("text", ""), image_path=tmp_path, url=f.get("url", ""),
                   channel=f.get("channel", "other"), payment_method=f.get("payment_method", "none"),
                   account_age_days=_num(f.get("account_age_days"), -1),
                   followers=_num(f.get("followers"), 0), following=_num(f.get("following"), 0),
                   verified=int(f.get("verified") == "on"), sender_in_contacts=int(f.get("in_contacts") == "on"))
        report = MODEL.explain(pd.DataFrame([row]))[0]
    except ValueError:
        return jsonify(error="Could not read that image. Please upload a PNG or JPG screenshot."), 400
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
    report["modalities"] = MODALITIES
    return jsonify(report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()
    load_model(a.model)
    print(f"model: {MODEL_PATH}")
    app.run(host="127.0.0.1", port=a.port, debug=False)
