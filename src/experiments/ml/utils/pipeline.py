"""Evaluation of combinations (encoder x classifier) with 80/10/10 hold-out.

Performs data splitting, injection and training of encoders
and classifiers, and extraction of performance metrics.
Main output: a DataFrame with one row per evaluated combination
and columns with detailed metrics (F1-Score, Accuracy, Times, etc.).

Typical usage:
    from src.experiments.ml.utils.data import load_all_datasets
    from src.experiments.ml.utils.pipeline import run_grid

    splits = load_all_datasets()
    df_results, conf_mats = run_grid(splits)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.experiments.ml.utils.classifiers import CLASSIFIER_NAMES, make_classifier
from src.experiments.ml.utils.data import CLASS_ORDER, Splits
from src.experiments.ml.utils.encoders import ENCODER_NAMES, make_encoder, _hash_corpus


SEED = 42


@dataclass
class EvalResult:
    dataset: str
    encoder: str
    classifier: str
    n_train: int
    n_val: int
    n_test: int
    val_macro_f1: float
    val_accuracy: float
    test_macro_f1: float
    test_accuracy: float
    test_f1_ham: float
    test_f1_spam: float
    test_f1_smishing: float
    fit_time_s: float
    predict_time_s: float


def _evaluate_one(
    encoder_name: str,
    classifier_name: str,
    X_train: list[str],
    y_train: list[str],
    X_val: list[str],
    y_val: list[str],
    X_test: list[str],
    y_test: list[str],
) -> tuple[EvalResult, np.ndarray]:
    """Executes a single fit + predict (on val and test) and returns metrics."""
    encoder = make_encoder(encoder_name)

    t0 = time.perf_counter()
    X_train_encoded = encoder.fit_transform(X_train)
    X_val_encoded = encoder.transform(X_val)
    X_test_encoded = encoder.transform(X_test)
    fit_time = time.perf_counter() - t0

    dense = encoder.output == "dense"
    clf = make_classifier(classifier_name, dense=dense)

    if dense and hasattr(X_train_encoded, "toarray"):
        X_train_encoded = X_train_encoded.toarray()
        X_val_encoded = X_val_encoded.toarray()
        X_test_encoded = X_test_encoded.toarray()

    le = LabelEncoder().fit(CLASS_ORDER)
    y_train_enc = le.transform(y_train)

    t1 = time.perf_counter()
    clf.fit(X_train_encoded, y_train_enc)
    fit_time += time.perf_counter() - t1

    t2 = time.perf_counter()
    # Predict on Validation
    y_val_pred_enc = clf.predict(X_val_encoded)
    y_val_pred = le.inverse_transform(np.asarray(y_val_pred_enc, dtype=int))
    val_macro = float(f1_score(y_val, y_val_pred, average="macro", labels=CLASS_ORDER, zero_division=0))
    val_acc = accuracy_score(y_val, y_val_pred)
    
    # Predict on Test
    y_test_pred_enc = clf.predict(X_test_encoded)
    predict_time = time.perf_counter() - t2

    y_test_pred = le.inverse_transform(np.asarray(y_test_pred_enc, dtype=int))

    test_macro = float(f1_score(y_test, y_test_pred, average="macro", labels=CLASS_ORDER, zero_division=0))
    test_acc = accuracy_score(y_test, y_test_pred)
    
    test_per_class = np.asarray(f1_score(y_test, y_test_pred, average=None, labels=CLASS_ORDER, zero_division=0))
    cm_test = confusion_matrix(y_test, y_test_pred, labels=CLASS_ORDER)

    return (
        EvalResult(
            dataset="",
            encoder=encoder_name,
            classifier=classifier_name,
            n_train=len(y_train),
            n_val=len(y_val),
            n_test=len(y_test),
            val_macro_f1=float(val_macro),
            val_accuracy=float(val_acc),
            test_macro_f1=float(test_macro),
            test_accuracy=float(test_acc),
            test_f1_ham=float(test_per_class[0]),
            test_f1_spam=float(test_per_class[1]),
            test_f1_smishing=float(test_per_class[2]),
            fit_time_s=float(fit_time),
            predict_time_s=float(predict_time),
        ),
        cm_test,
    )


def run_grid(
    splits: dict[str, Splits],
    encoders: Iterable[str] = ENCODER_NAMES,
    classifiers: Iterable[str] = CLASSIFIER_NAMES,
    seed: int = SEED,
    verbose: bool = True,
    completed: set[tuple[str, str, str]] | None = None,
) -> Iterator[tuple[EvalResult, np.ndarray]]:
    """Executes the complete evaluation grid with 80/10/10 Hold-Out.
    Yields (EvalResult, confusion_matrix) on the fly for checkpointing.
    """
    if completed is None:
        completed = set()

    for ds_key, sp in splits.items():
        texts = sp.texts
        labels = np.array(sp.labels)
        full_corpus = list(texts)
        cache_key = f"{ds_key}_{_hash_corpus(full_corpus)}"

        # Check if entire dataset is completed
        if all((sp.name, enc, clf) in completed for enc in encoders for clf in classifiers):
            if verbose:
                print(f"Skipping dataset {sp.name} (all combinations already done)", flush=True)
            continue

        # Split 80/10/10
        # 1. Split 80% train / 20% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            full_corpus, labels, test_size=0.2, random_state=seed, stratify=labels
        )
        # 2. Split 20% temp into 10% val / 10% test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
        )

        for enc_name in encoders:
            # check if ALL classifiers for this encoder are already completed
            if all((sp.name, enc_name, clf_name) in completed for clf_name in classifiers):
                if verbose:
                    print(f"Skipping encoder {enc_name} on {sp.name} (all classifiers already done)", flush=True)
                continue



            for clf_name in classifiers:
                if (sp.name, enc_name, clf_name) in completed:
                    if verbose:
                        print(f"Skipping {sp.name} {enc_name} × {clf_name} (already done)", flush=True)
                    continue

                if verbose:
                    print(
                        f"[{sp.name}] {enc_name} × {clf_name} "
                        f"(train={len(X_train)}, val={len(X_val)}, test={len(X_test)})",
                        flush=True,
                    )

                res, cm = _evaluate_one(
                    encoder_name=enc_name,
                    classifier_name=clf_name,
                    X_train=X_train,
                    y_train=y_train.tolist(),
                    X_val=X_val,
                    y_val=y_val.tolist(),
                    X_test=X_test,
                    y_test=y_test.tolist(),
                )
                res.dataset = sp.name
                yield res, cm


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """For compatibility or in case results need to be sorted.
    Since it is a single pass (80/10/10), there are no means or deviations.
    We simply sort by test_macro_f1."""
    return df.sort_values(["dataset", "test_macro_f1"], ascending=[True, False])

