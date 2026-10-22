"""Optional neural fusion: per-modality encoders + masked attention over modalities.

The attention weights tell you which modality drove each decision.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from .fusion import MultimodalScamDetector, MODALITIES
from .image_features import N_IMAGE_FEATURES
from .text_features import TextFeaturizer


class AttentionFusionNet(nn.Module):
    def __init__(self, dims, hidden=64, dropout=0.2):
        super().__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
            for d in dims])
        self.modality_emb = nn.Parameter(torch.randn(len(dims), hidden) * 0.02)
        self.score = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, xs, mask):
        h = torch.stack([enc(x) for enc, x in zip(self.encoders, xs)], 1) + self.modality_emb
        s = self.score(torch.tanh(h)).squeeze(-1).masked_fill(~mask, -1e9)
        w = torch.softmax(s, 1)
        return self.head((w.unsqueeze(-1) * h).sum(1)).squeeze(-1), w


class NeuralScamDetector(MultimodalScamDetector):
    """Re-uses the feature extractors of the stacking model, swaps the fusion for a neural net."""

    def __init__(self, epochs=60, lr=2e-3, svd_dim=128, random_state=42, use_ocr=False):
        super().__init__(use_ocr=use_ocr, random_state=random_state)
        self.epochs, self.lr, self.svd_dim = epochs, lr, svd_dim

    def _tensors(self, df, fit=False):
        texts, has_text, _, has_img, img_X, meta_X, _ = self._extract(df)
        if fit:
            self.tf_ = TextFeaturizer().fit(texts)
            T = self.tf_.transform(texts)
            self.svd_ = TruncatedSVD(min(self.svd_dim, T.shape[1] - 1), random_state=self.random_state).fit(T)
            self.tsc_ = StandardScaler().fit(self.svd_.transform(T))
            self.isc_ = StandardScaler().fit(img_X if img_X is not None else np.zeros((2, N_IMAGE_FEATURES)))
            self.msc_ = StandardScaler().fit(meta_X)
        t = self.tsc_.transform(self.svd_.transform(self.tf_.transform(texts)))
        im = np.zeros((len(df), N_IMAGE_FEATURES))
        if img_X is not None:
            im[has_img] = self.isc_.transform(img_X)
        m = self.msc_.transform(meta_X)
        mask = np.column_stack([has_text, has_img, np.ones(len(df), bool)])
        f = lambda a: torch.tensor(a, dtype=torch.float32)
        return [f(t), f(im), f(m)], torch.tensor(mask)

    def fit(self, df, y, groups=None):
        torch.manual_seed(self.random_state)
        xs, mask = self._tensors(df, fit=True)
        y = torch.tensor(np.asarray(y), dtype=torch.float32)
        self.net_ = AttentionFusionNet([x.shape[1] for x in xs])
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        n = len(y)
        for _ in range(self.epochs):
            self.net_.train()
            for idx in torch.randperm(n).split(64):
                m = mask[idx].clone()
                drop = torch.rand(m.shape) < 0.25
                drop[:, 2] = False
                m &= ~drop  # modality dropout
                logits, _ = self.net_([x[idx] for x in xs], m)
                loss = loss_fn(logits, y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict_proba(self, df):
        return self.predict_with_attention(df)[0]

    @torch.no_grad()
    def predict_with_attention(self, df):
        self.net_.eval()
        xs, mask = self._tensors(df)
        logits, w = self.net_(xs, mask)
        return torch.sigmoid(logits).numpy(), [dict(zip(MODALITIES, r.round(decimals=3).tolist())) for r in w]
