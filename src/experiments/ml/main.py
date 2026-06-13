"""Entry point for the Master's Thesis (TFM) Machine Learning pipeline.

Executes the evaluation of different encoders and classifiers on the
consolidated MetaSMS dataset (400,000 messages). Uses a stratified
80% (train) / 10% (validation) / 10% (test) partition and saves the
detailed results in `output/ml/`.

Typical usage from the project root:

    python -m src.experiments.ml.main --data_root data/processed --out_dir output/ml

Relevant arguments:
    --datasets    Subset of datasets to evaluate (default: metasms).
    --encoders    Subset of encoders (default: bow tfidf w2v fasttext).
    --classifiers Subset of classifiers (default: nb logreg rf svm).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# FIX: Import torch before pandas/gensim to avoid DLL conflicts on Windows
# Wrap in try-except so it doesn't crash on the Legio Linux cluster where we omit heavy PyTorch
try:
    import torch
except ImportError:
    pass

import pandas as pd

from src.experiments.ml.utils.classifiers import CLASSIFIER_NAMES
from src.experiments.ml.utils.data import CLASS_ORDER, load_all_datasets, summarize
from src.experiments.ml.utils.encoders import ENCODER_NAMES
from src.experiments.ml.utils.pipeline import aggregate, run_grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ML Pipeline 4x4 with 80/10/10 Hold-out")
    p.add_argument("--data_root", type=str, default="data/processed")
    p.add_argument("--out_dir", type=str, default="output/ml")
    p.add_argument("--datasets", nargs="+", default=["metasms"])
    p.add_argument("--encoders", nargs="+", default=ENCODER_NAMES)
    p.add_argument("--classifiers", nargs="+", default=CLASSIFIER_NAMES)
    p.add_argument("--feature_type", type=str, default="norm", choices=["norm", "raw", "anonymized", "injected"], help="Text representation to use")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f">> Loading datasets (Feature View: {args.feature_type})...", flush=True)
    splits_all = load_all_datasets(args.data_root, feature_type=args.feature_type)

    unknown = [k for k in args.datasets if k not in splits_all]
    if unknown:
        print(
            "!! Unknown datasets (skipped): "
            + ", ".join(repr(k) for k in unknown)
            + f". Valid values: {', '.join(sorted(splits_all))}",
            flush=True,
        )

    splits = {k: splits_all[k] for k in args.datasets if k in splits_all}

    summary = summarize(splits)
    print(summary)
    summary.to_csv(out_dir / "datasets_summary.csv")

    completed_set = set()
    results_file = out_dir / "results.csv"
    cm_file = out_dir / "confusion_matrices.csv"

    if results_file.exists():
        try:
            df_prev = pd.read_csv(results_file)
            for _, row in df_prev.iterrows():
                completed_set.add((row["dataset"], row["encoder"], row["classifier"]))
            print(f">> Found existing results.csv. Skipping {len(completed_set)} completed fits.")
        except Exception as e:
            print(f"!! Failed to read existing results.csv: {e}")

    print(
        f">> Launching grid: {len(splits)} datasets x "
        f"{len(args.encoders)} encoders x {len(args.classifiers)} classifiers = "
        f"{len(splits) * len(args.encoders) * len(args.classifiers)} fits.",
        flush=True,
    )

    meta = {
        "split": "80/10/10",
        "seed": 42,
        "encoders": list(args.encoders),
        "classifiers": list(args.classifiers),
        "datasets": list(args.datasets),
        "class_order": CLASS_ORDER,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for res, cm in run_grid(
        splits=splits,
        encoders=args.encoders,
        classifiers=args.classifiers,
        verbose=not args.quiet,
        completed=completed_set,
    ):
        # Save result
        df_res = pd.DataFrame([res.__dict__])
        has_res_headers = results_file.exists() and results_file.stat().st_size > 0
        df_res.to_csv(results_file, mode="a", index=False, header=not has_res_headers)
        
        # Save CM
        cm_records = []
        for i, true_lbl in enumerate(CLASS_ORDER):
            for j, pred_lbl in enumerate(CLASS_ORDER):
                cm_records.append({
                    "dataset": res.dataset,
                    "encoder": res.encoder,
                    "classifier": res.classifier,
                    "true_label": true_lbl,
                    "predicted_label": pred_lbl,
                    "count": int(cm[i, j]),
                })
        df_cm = pd.DataFrame(cm_records)
        has_cm_headers = cm_file.exists() and cm_file.stat().st_size > 0
        df_cm.to_csv(cm_file, mode="a", index=False, header=not has_cm_headers)

    print(f">> Grid finished.", flush=True)

    if results_file.exists():
        df_results = pd.read_csv(results_file)
        df_agg = aggregate(df_results)
        df_agg.to_csv(out_dir / "summary.csv", index=False)

        print(">> Final summary (top-3 per dataset by test_macro_f1):", flush=True)
        for ds_name, sub in df_agg.groupby("dataset"):
            print(f"\n=== {ds_name} ===")
            print(sub.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
