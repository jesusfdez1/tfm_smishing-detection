"""Entry point for the Deep Learning (SLM) pipeline.

Evaluates multiple Small Language Models (BETO, MiniLM, etc.)
using a strict 80/10/10 hold-out split on the MetaSMS dataset.

Usage:
    python -m src.experiments.dl.main --models minilm beto
"""

import argparse
import json
import pandas as pd
from pathlib import Path

from src.experiments.ml.utils.data import load_all_datasets, CLASS_ORDER
from src.experiments.dl.utils.models import SLM_MODELS, get_model_name
from src.experiments.dl.utils.pipeline import run_slm_evaluation

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DL Pipeline with 80/10/10 Hold-out")
    p.add_argument("--data_root", type=str, default="data/processed")
    p.add_argument("--out_dir", type=str, default="output/dl")
    p.add_argument("--models", nargs="+", default=list(SLM_MODELS.keys()), help="Model keys to evaluate")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size for training/eval")
    p.add_argument("--epochs", type=int, default=3, help="Max number of epochs")
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    p.add_argument("--feature_type", type=str, default="norm", choices=["norm", "raw", "anonymized", "injected"], help="Text representation to use")
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f">> Loading datasets for DL (Feature View: {args.feature_type})...", flush=True)
    splits_all = load_all_datasets(args.data_root, feature_type=args.feature_type)
    metasms = splits_all.get("metasms")
    if not metasms:
        raise ValueError("MetaSMS dataset not found!")

    texts = metasms.texts
    labels = metasms.labels
    
    results_file = out_dir / "results.csv"
    cm_file = out_dir / "confusion_matrices.csv"
    
    completed_set = set()
    if results_file.exists():
        try:
            df_prev = pd.read_csv(results_file)
            for _, row in df_prev.iterrows():
                completed_set.add(row["model_key"])
            print(f">> Found existing results.csv. Skipping {len(completed_set)} completed models.")
        except Exception as e:
            print(f"!! Failed to read existing results.csv: {e}")

    meta = {
        "split": "80/10/10",
        "models_target": list(args.models),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f">> Launching DL evaluation for {len(args.models)} models...", flush=True)
    
    for model_key in args.models:
        if model_key in completed_set:
            print(f"Skipping {model_key} (already completed)", flush=True)
            continue
            
        hf_id = get_model_name(model_key)
        print(f"\n=======================================================", flush=True)
        print(f"=== Evaluating {model_key} ({hf_id}) ===", flush=True)
        print(f"=======================================================", flush=True)
        
        try:
            res_dict = run_slm_evaluation(
                model_name=model_key,
                hf_model_id=hf_id,
                texts=texts,
                labels=labels,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.lr,
                out_dir=f"{args.out_dir}/checkpoints"
            )
            
            # Extract CM before saving
            cm = res_dict.pop("confusion_matrix")
            
            # Save metrics
            df_res = pd.DataFrame([res_dict])
            has_res_headers = results_file.exists() and results_file.stat().st_size > 0
            df_res.to_csv(results_file, mode="a", index=False, header=not has_res_headers)
            
            # Save CM
            cm_records = []
            for i, true_lbl in enumerate(CLASS_ORDER):
                for j, pred_lbl in enumerate(CLASS_ORDER):
                    cm_records.append({
                        "dataset": "MetaSMS",
                        "model_key": model_key,
                        "true_label": true_lbl,
                        "predicted_label": pred_lbl,
                        "count": int(cm[i, j])
                    })
            df_cm = pd.DataFrame(cm_records)
            has_cm_headers = cm_file.exists() and cm_file.stat().st_size > 0
            df_cm.to_csv(cm_file, mode="a", index=False, header=not has_cm_headers)
            
            print(f">> OK: {model_key} evaluation saved.", flush=True)
        except Exception as e:
            print(f"!! ERROR evaluating {model_key}: {e}", flush=True)

    print(">> Deep Learning pipeline finished.", flush=True)

if __name__ == "__main__":
    main()
