"""Baseline: handcrafted numeric features + TF-IDF, LightGBM 3-class classifier.

Runs stratified K-fold CV (reporting log loss, the competition metric), then
retrains on all data and writes submissions/submission.csv for the test set.

Usage:
    ./.venv/bin/python src/baseline.py            # full run + submission
    ./.venv/bin/python src/baseline.py --folds 5  # customise CV
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from data import SUB_DIR, TARGETS, load_test, load_train, target_labels
from features import build_features

SEED = 42


def _tfidf_inputs(df):
    """Concatenated prompt+A and prompt+B text for a shared TF-IDF vocabulary."""
    a = (df["prompt_text"] + " " + df["response_a_text"]).tolist()
    b = (df["prompt_text"] + " " + df["response_b_text"]).tolist()
    return a, b


def build_matrix(df, vec: TfidfVectorizer, fit: bool):
    num = build_features(df).values
    a, b = _tfidf_inputs(df)
    if fit:
        vec.fit(a + b)
    Xa = vec.transform(a)
    Xb = vec.transform(b)
    # Symmetric-ish representation: A, B, and their difference.
    return sp.hstack([sp.csr_matrix(num), Xa, Xb, Xa - Xb], format="csr")


def lgb_params() -> dict:
    return dict(
        objective="multiclass",
        num_class=3,
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.6,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_child_samples=50,
        seed=SEED,
        verbose=-1,
    )


def run(folds: int, rounds: int, quick: int | None) -> None:
    import lightgbm as lgb

    t0 = time.time()
    train = load_train()
    if quick:
        train = train.sample(quick, random_state=SEED).reset_index(drop=True)
        print(f"[quick] using {quick} rows")
    y = target_labels(train).values
    print(f"train rows: {len(train)}  | class balance: {np.bincount(y) / len(y)}")

    vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X = build_matrix(train, vec, fit=True)
    print(f"feature matrix: {X.shape}  (built in {time.time() - t0:.1f}s)")

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), 3))
    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t_fold = time.time()
        dtr = lgb.Dataset(X[tr_idx], y[tr_idx])
        dva = lgb.Dataset(X[va_idx], y[va_idx])
        model = lgb.train(
            lgb_params(),
            dtr,
            num_boost_round=rounds,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va_idx] = model.predict(X[va_idx])
        best_iters.append(model.best_iteration)
        fold_ll = log_loss(y[va_idx], oof[va_idx], labels=[0, 1, 2])
        print(f"  fold {fold}: log_loss={fold_ll:.4f}  best_iter={model.best_iteration}  ({time.time() - t_fold:.0f}s)")

    cv = log_loss(y, oof, labels=[0, 1, 2])
    print(f"\nOOF log_loss: {cv:.4f}   (uniform 1/3 baseline = {np.log(3):.4f})")

    # Retrain on all data using the CV's best-iteration count (with a small buffer
    # for the extra ~20% of training data), instead of an expensive fixed 3000 rounds.
    final_rounds = int(np.median(best_iters) * folds / (folds - 1)) + 1
    print(f"retraining on full data for submission ({final_rounds} rounds)...")
    full = lgb.train(lgb_params(), lgb.Dataset(X, y), num_boost_round=final_rounds)
    test = load_test()
    Xte = build_matrix(test, vec, fit=False)
    preds = full.predict(Xte)

    sub = test[["id"]].copy()
    sub[TARGETS] = preds
    out = SUB_DIR / "submission.csv"
    sub.to_csv(out, index=False)
    print(f"wrote {out}  ({len(sub)} rows)  | total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--quick", type=int, default=None, help="subsample N rows for a fast smoke test")
    args = ap.parse_args()
    run(args.folds, args.rounds, args.quick)
