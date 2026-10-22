"""Text modality: scam-pattern lexicon + TF-IDF."""
import re

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

# Each category is a known investment-scam tactic. Patterns are case-insensitive.
SCAM_PATTERNS = {
    "guaranteed_returns": [
        r"guarantee[d]?\s+(returns?|profits?|income)", r"risk[\s-]?free", r"no\s+risk",
        r"100\s?%\s+(safe|secure|guaranteed)", r"zero\s+loss", r"assured\s+(returns?|profits?)",
        r"(returns?|profits?|payouts?|income)\s+(are\s+)?(guaranteed|assured)", r"principal\s+guaranteed",
    ],
    "unrealistic_returns": [
        r"double\s+your\s+(money|investment)", r"\b\d{1,3}x\s+(returns?|profits?)",
        r"paisa\s+double", r"\d{2,4}\s?%\s*(daily|weekly|monthly|per\s+(day|week|month)|in\s+\d+\s+(days?|weeks?))",
        r"\d{2,4}\s?%\s+(daily|weekly|monthly)?\s*(returns?|profits?|gains?)\s+(daily|weekly|monthly|every)",
        r"\d{2,4}\s?%\s+(monthly|weekly|daily)\s+(returns?|profits?|gains?)", r"\d{3,4}\s?%\s+(apy|returns?)",
        r"earn\s+(rs\.?|₹|\$)?\s?\d[\d,]*\s+(daily|per\s+day|every\s+day)",
    ],
    "urgency": [
        r"limited\s+(slots?|seats?|time|offer)", r"act\s+(now|fast)", r"only\s+today",
        r"hurry", r"last\s+chance", r"expires?\s+(today|soon|in)", r"before\s+it'?s\s+too\s+late",
    ],
    "off_platform": [
        r"telegram", r"whats\s?app", r"\bdm\s+me\b", r"join\s+(my|our)\s+(vip\s+)?(group|channel)",
        r"t\.me/", r"wa\.me/", r"inbox\s+me", r"signal\s+app",
    ],
    "crypto_forex": [
        r"\busdt\b", r"bitcoin|\bbtc\b", r"\bforex\b", r"binary\s+options?", r"crypto\s+mining",
        r"airdrop", r"trading\s+bot", r"\bpresale\b", r"new\s+coin",
    ],
    "payment_red_flags": [
        r"gift\s?cards?", r"registration\s+fee", r"deposit\s+first", r"wallet\s+address",
        r"send\s+(money|funds|payment)\s+to", r"pay\s+(via|through)\s+upi", r"activation\s+(fee|charge)",
    ],
    "withdrawal_trap": [
        r"withdraw(al)?\s+(fee|tax|charge)", r"unlock\s+(your\s+)?(funds|account|profits?)",
        r"pay\s+tax\s+to\s+withdraw", r"account\s+(frozen|locked)",
    ],
    "fake_authority": [
        r"sebi\s+(registered|approved)", r"insider\s+(tip|info)", r"vip\s+signals?",
        r"(my|our)\s+students?\s+(earned|made)", r"expert\s+(mentor|trader|analyst)",
        r"government\s+approved", r"rbi\s+approved",
    ],
    "relationship_lure": [
        r"\bdear\b", r"\bsweetheart\b", r"my\s+uncle", r"trust\s+me", r"i\s+will\s+teach\s+you",
        r"wrong\s+number",
    ],
}
_COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in SCAM_PATTERNS.items()}
PATTERN_NAMES = list(SCAM_PATTERNS)
_PCT = re.compile(r"(\d{1,4}(?:\.\d+)?)\s?%")


def find_patterns(text):
    """Return {category: [matched snippets]} for explainability."""
    hits = {}
    for cat, regs in _COMPILED.items():
        found = [m.group(0) for r in regs for m in r.finditer(text or "")]
        if found:
            hits[cat] = found
    return hits


def pattern_vector(text):
    text = text or ""
    counts = [sum(len(r.findall(text)) for r in regs) for regs in _COMPILED.values()]
    pcts = [float(p) for p in _PCT.findall(text)]
    max_pct = max(pcts) if pcts else 0.0
    extra = [
        np.log1p(max_pct),
        text.count("!") / (len(text) + 1) * 100,
        sum(c.isupper() for c in text) / (len(text) + 1),
        len(re.findall(r"[\U0001F300-\U0001FAFF]", text)),  # emoji count
        np.log1p(len(text)),
    ]
    return np.array(counts + extra, dtype=float)


class TextFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=20000):
        self.max_features = max_features

    def fit(self, texts, y=None):
        texts = [t or "" for t in texts]
        self.word_ = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=self.max_features,
                                     sublinear_tf=True)
        self.char_ = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                     max_features=self.max_features, sublinear_tf=True)
        self.word_.fit(texts)
        self.char_.fit(texts)
        return self

    def transform(self, texts):
        texts = [t or "" for t in texts]
        pats = np.vstack([pattern_vector(t) for t in texts])
        return hstack([self.word_.transform(texts), self.char_.transform(texts),
                       csr_matrix(pats * 0.5)]).tocsr()
