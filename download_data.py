"""Download the public datasets used by prepare_real_data.py into data/raw/.

Only data files are fetched (no code, no pickles). Usage:  python download_data.py
"""
import os
import urllib.request

HF = "https://huggingface.co/datasets"
FILES = {
    "shade63/sentinel_dataset_multiclass.csv":
        f"{HF}/Shade63/scam-classification-multiclass/resolve/main/sentinel_dataset_multiclass.csv",
    "shade63/README.md": f"{HF}/Shade63/scam-classification-multiclass/resolve/main/README.md",
    "hinglish/data.json": f"{HF}/bolewara/hinglish-scam-text-dataset/resolve/main/data.json",
    "hinglish/README.md": f"{HF}/bolewara/hinglish-scam-text-dataset/resolve/main/README.md",
    "cryptoscamdb/urls.yaml": "https://raw.githubusercontent.com/CryptoScamDB/blacklist/master/data/urls.yaml",
    "openphish/feed.txt": "https://openphish.com/feed.txt",
    "tranco/top-1m.csv.zip": "https://tranco-list.eu/top-1m.csv.zip",
}


def main():
    for rel, url in FILES.items():
        dest = os.path.join("data", "raw", rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest):
            print(f"skip   {rel} (exists)")
            continue
        print(f"fetch  {rel}")
        req = urllib.request.Request(url, headers={"User-Agent": "scam-detector-data-fetch"})
        with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
            f.write(r.read())
    for d in ("scam", "legit"):
        os.makedirs(os.path.join("data", "raw", "images", d), exist_ok=True)
    print("done. Add your own screenshots to data/raw/images/{scam,legit}/ then run prepare_real_data.py")


if __name__ == "__main__":
    main()
