"""Late-fusion (stacking) multimodal scam detector.

Each modality gets its own expert model. Their out-of-fold probabilities are
combined by a meta-learner that is trained with random modality dropout, so the
detector still works when a sample has no image, no text, etc.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .image_features import dhash, hamming, image_vector, ocr_text
from .meta_features import meta_flags, meta_vector
from .text_features import TextFeaturizer, find_patterns

MODALITIES = ["text", "image", "meta"]


def _has_image(p):
    return isinstance(p, str) and p.strip() != "" and os.path.exists(p)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class MultimodalScamDetector:
    def __init__(self, use_ocr=False, n_folds=5, random_state=42):
        self.use_ocr = use_ocr
        self.n_folds = n_folds
        self.random_state = random_state

    # ---------- feature extraction ----------
    def _texts(self, df):
        texts = df.get("text", pd.Series([""] * len(df))).fillna("").astype(str).tolist()
        if self.use_ocr and "image_path" in df:
            texts = [t + " " + (ocr_text(p) if _has_image(p) else "")
                     for t, p in zip(texts, df["image_path"])]
        return texts

    def _extract(self, df):
        df = df.reset_index(drop=True)
        texts = self._texts(df)
        img_paths = df.get("image_path", pd.Series([None] * len(df))).tolist()
        has_text = np.array([len(t.strip()) > 0 for t in texts])
        has_img = np.array([_has_image(p) for p in img_paths])
        img_X = np.vstack([image_vector(p) for p, h in zip(img_paths, has_img) if h]) if has_img.any() else None
        meta_X = np.vstack([meta_vector(r) for r in df.to_dict("records")])
        return texts, has_text, img_paths, has_img, img_X, meta_X

    # ---------- training ----------
    def fit(self, df, y, groups=None):
        """groups: optional template ids, so out-of-fold scores aren't inflated by near-duplicates."""
        y = np.asarray(y).astype(int)
        texts, has_text, img_paths, has_img, img_X, meta_X = self._extract(df)
        g = None if groups is None else np.asarray(groups)
        if g is None:
            splitter = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.random_state)
            cv = lambda idx: splitter
        else:
            splitter = StratifiedGroupKFold(self.n_folds, shuffle=True, random_state=self.random_state)
            cv = lambda idx: list(splitter.split(idx, y[idx], g[idx]))
        oof = np.full((len(y), 3), 0.5)

        self.text_model_ = make_pipeline(TextFeaturizer(),
                                         LogisticRegression(C=4, max_iter=2000, class_weight="balanced"))
        t_idx = np.nonzero(has_text)[0]
        tx = [texts[i] for i in t_idx]
        oof[t_idx, 0] = cross_val_predict(self.text_model_, tx, y[t_idx], cv=cv(t_idx), method="predict_proba")[:, 1]
        self.text_model_.fit(tx, y[t_idx])

        self.image_model_ = None
        if has_img.sum() >= 2 * self.n_folds:
            self.image_model_ = make_pipeline(StandardScaler(), RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=self.random_state))
            i_idx = np.nonzero(has_img)[0]
            oof[i_idx, 1] = cross_val_predict(self.image_model_, img_X, y[i_idx], cv=cv(i_idx),
                                              method="predict_proba")[:, 1]
            self.image_model_.fit(img_X, y[i_idx])
            self.scam_hashes_ = [dhash(img_paths[i]) for i in i_idx if y[i] == 1]
        else:
            self.scam_hashes_ = []

        self.meta_model_ = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                          random_state=self.random_state)
        oof[:, 2] = cross_val_predict(self.meta_model_, meta_X, y, cv=cv(np.arange(len(y))), method="predict_proba")[:, 1]
        self.meta_model_.fit(meta_X, y)

        present = np.column_stack([has_text, has_img, np.ones(len(y), bool)])
        Z, yz = self._augment(oof, present, y)
        self.fuser_ = LogisticRegression(max_iter=1000).fit(Z, yz)
        self.oof_ = oof
        return self

    def _stack(self, probs, present):
        p = np.where(present, probs, 0.5)
        return np.hstack([_logit(p) * present, (~present).astype(float)])

    def _augment(self, oof, present, y, copies=3):
        """Modality dropout: teach the fuser to cope with missing inputs."""
        rng = np.random.default_rng(self.random_state)
        Zs, ys = [self._stack(oof, present)], [y]
        for _ in range(copies):
            drop = rng.random(present.shape) < 0.3
            drop[:, 2] = False  # metadata is always available
            Zs.append(self._stack(oof, present & ~drop))
            ys.append(y)
        return np.vstack(Zs), np.concatenate(ys)

    # ---------- inference ----------
    def modality_probs(self, df):
        texts, has_text, img_paths, has_img, img_X, meta_X = self._extract(df)
        probs = np.full((len(df), 3), 0.5)
        if has_text.any():
            probs[has_text, 0] = self.text_model_.predict_proba([t for t, h in zip(texts, has_text) if h])[:, 1]
        if self.image_model_ is not None and has_img.any():
            probs[has_img, 1] = self.image_model_.predict_proba(img_X)[:, 1]
        else:
            has_img = np.zeros(len(df), bool)
        probs[:, 2] = self.meta_model_.predict_proba(meta_X)[:, 1]
        present = np.column_stack([has_text, has_img, np.ones(len(df), bool)])
        return probs, present, texts, img_paths

    def predict_proba(self, df):
        probs, present, _, _ = self.modality_probs(df)
        return self.fuser_.predict_proba(self._stack(probs, present))[:, 1]

    def predict(self, df, threshold=0.5):
        return (self.predict_proba(df) >= threshold).astype(int)

    def explain(self, df, threshold=0.5):
        """Per-sample report: fused score, per-modality scores, and the patterns that fired."""
        df = df.reset_index(drop=True)
        probs, present, texts, img_paths = self.modality_probs(df)
        fused = self.fuser_.predict_proba(self._stack(probs, present))[:, 1]
        out = []
        for i, row in enumerate(df.to_dict("records")):
            rep = {
                "scam_probability": round(float(fused[i]), 4),
                "verdict": "LIKELY SCAM" if fused[i] >= threshold else "likely legitimate",
                "modality_scores": {m: (round(float(probs[i, j]), 4) if present[i, j] else None)
                                    for j, m in enumerate(MODALITIES)},
                "text_patterns": find_patterns(texts[i]),
                "metadata_flags": meta_flags(row),
            }
            if present[i, 1] and self.scam_hashes_:
                h = dhash(img_paths[i])
                d = min(hamming(h, s) for s in self.scam_hashes_)
                if d <= 6:
                    rep["image_template_match"] = f"near-duplicate of a known scam image (hamming={d})"
            out.append(rep)
        return out

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
