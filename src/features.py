"""Handcrafted numeric features for the preference-prediction baseline.

The design goal is to capture the signals known to drive Arena preferences:
length / verbosity, formatting (markdown, code, lists), and the *difference*
between response A and response B (since the target is which one wins).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_CODE = re.compile(r"```")
_LIST = re.compile(r"(?m)^\s*(?:[-*+]|\d+\.)\s")
_HEADER = re.compile(r"(?m)^#{1,6}\s")


def _text_stats(text: str) -> dict[str, float]:
    n_char = len(text)
    words = text.split()
    n_word = len(words)
    return {
        "n_char": n_char,
        "n_word": n_word,
        "n_line": text.count("\n") + 1,
        "n_code": len(_CODE.findall(text)) // 2,
        "n_list": len(_LIST.findall(text)),
        "n_header": len(_HEADER.findall(text)),
        "avg_word_len": (sum(len(w) for w in words) / n_word) if n_word else 0.0,
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dense DataFrame of numeric features (one row per battle)."""
    rows = []
    for prompt, a, b in zip(df["prompt_text"], df["response_a_text"], df["response_b_text"]):
        pa = _text_stats(a)
        pb = _text_stats(b)
        pp = _text_stats(prompt)
        feat: dict[str, float] = {}
        for k, v in pp.items():
            feat[f"prompt_{k}"] = v
        for k in pa:
            feat[f"a_{k}"] = pa[k]
            feat[f"b_{k}"] = pb[k]
            feat[f"diff_{k}"] = pa[k] - pb[k]
            feat[f"absdiff_{k}"] = abs(pa[k] - pb[k])
            denom = pa[k] + pb[k]
            feat[f"ratio_{k}"] = (pa[k] - pb[k]) / denom if denom else 0.0
        feat["a_empty"] = float(len(a.strip()) == 0)
        feat["b_empty"] = float(len(b.strip()) == 0)
        rows.append(feat)
    return pd.DataFrame(rows, index=df.index).astype(np.float32)


if __name__ == "__main__":
    from data import load_train

    tr = load_train().head(1000)
    X = build_features(tr)
    print("feature matrix:", X.shape)
    print(list(X.columns))
