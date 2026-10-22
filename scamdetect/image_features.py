"""Image modality: visual patterns typical of scam promos / fake profit screenshots.

Handcrafted features (no downloads needed):
  * colour statistics (green "profit" dominance, saturation, brightness)
  * edge / text density (screenshots full of big numbers)
  * upward-trend score: slope of green pixels across the image (fake rising charts)
  * perceptual hash, for matching known scam templates
  * optional OCR (pytesseract) whose text is fed into the text-pattern lexicon
"""
import cv2
import numpy as np

SIZE = 256
N_IMAGE_FEATURES = 18 + 8 + 8 + 12


def load_image(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)  # unicode-safe on Windows
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.resize(img, (SIZE, SIZE))


def _trend_score(mask):
    """Slope of the mask's pixel centroid per column; >0 means rising left->right."""
    cols, ys = [], []
    for x in range(0, SIZE, 4):
        idx = np.nonzero(mask[:, x])[0]
        if len(idx):
            cols.append(x)
            ys.append(idx.mean())
    if len(cols) < 8:
        return 0.0
    slope = np.polyfit(cols, ys, 1)[0]
    return float(-slope)  # image y grows downward


def image_vector(path):
    img = load_image(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    n = h.size

    hue_hist = np.histogram(h, bins=18, range=(0, 180))[0] / n
    sat_hist = np.histogram(s, bins=8, range=(0, 256))[0] / n
    val_hist = np.histogram(v, bins=8, range=(0, 256))[0] / n

    colourful = s > 80
    green = ((h > 40) & (h < 85) & colourful)
    red = (((h < 10) | (h > 170)) & colourful)
    gold = ((h > 15) & (h < 35) & colourful)
    edges = cv2.Canny(gray, 80, 200)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    _, text_mask = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)))
    n_blobs = cv2.connectedComponents(text_mask)[0]

    extra = np.array([
        green.mean(), red.mean(), gold.mean(),
        green.mean() / (red.mean() + 1e-3) / 100,
        edges.mean() / 255, text_mask.mean() / 255, np.log1p(n_blobs) / 10,
        gray.mean() / 255, gray.std() / 128, s.mean() / 255,
        _trend_score(green) , _trend_score(red),
    ])
    return np.concatenate([hue_hist, sat_hist, val_hist, extra]).astype(float)


def dhash(path, size=8):
    g = cv2.cvtColor(load_image(path), cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size + 1, size))
    return (g[:, 1:] > g[:, :-1]).flatten()


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def ocr_text(path):
    """Extract text if pytesseract + Tesseract are installed; otherwise ''."""
    try:
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(path))
    except Exception:
        return ""
