"""Data loading and text parsing for the LLM Classification Finetuning competition.

The `prompt`, `response_a`, and `response_b` columns are stored as JSON-encoded
lists of strings (one entry per conversation turn). We parse them into plain
Python lists and expose helpers for joining turns into a single string.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUB_DIR = ROOT / "submissions"

TARGETS = ["winner_model_a", "winner_model_b", "winner_tie"]


def _parse_list(x: str) -> list[str]:
    """Parse a JSON-encoded list of strings; be tolerant of malformed rows."""
    if isinstance(x, list):
        return [str(t) for t in x]
    if not isinstance(x, str):
        return [""]
    try:
        val = json.loads(x)
    except (json.JSONDecodeError, TypeError):
        # Some rows contain literal 'null' entries or bad escapes; fall back to raw.
        return [x]
    if isinstance(val, list):
        return ["" if t is None else str(t) for t in val]
    return ["" if val is None else str(val)]


def load(name: str) -> pd.DataFrame:
    """Load train.csv or test.csv and parse the text list columns.

    Adds `<col>_list` (list[str]) and `<col>_text` (turns joined by newlines).
    """
    df = pd.read_csv(DATA_DIR / name)
    for col in ["prompt", "response_a", "response_b"]:
        if col in df.columns:
            df[f"{col}_list"] = df[col].map(_parse_list)
            df[f"{col}_text"] = df[f"{col}_list"].map(lambda ts: "\n".join(ts))
    return df


def load_train() -> pd.DataFrame:
    return load("train.csv")


def load_test() -> pd.DataFrame:
    return load("test.csv")


def target_labels(df: pd.DataFrame) -> pd.Series:
    """Collapse the three one-hot target columns into a single 0/1/2 label.

    0 = model_a wins, 1 = model_b wins, 2 = tie.
    """
    import numpy as np

    return pd.Series(np.argmax(df[TARGETS].values, axis=1), index=df.index, name="label")


if __name__ == "__main__":
    tr = load_train()
    print("train:", tr.shape)
    print(tr[["id", "model_a", "model_b"] + TARGETS].head())
    print("label distribution:")
    print(target_labels(tr).value_counts(normalize=True).rename({0: "a", 1: "b", 2: "tie"}))
