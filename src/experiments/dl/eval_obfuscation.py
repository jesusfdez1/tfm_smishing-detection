"""Entry point for the Obfuscation Robustness evaluation.

Loads a previously trained checkpoint from `output/dl/checkpoints`
and evaluates its performance exclusively on the subset of the test split
that contains intentional obfuscation (e.g. greek letters, character substitution).
"""

import argparse
import os
import json
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from src.experiments.ml.utils.data import CLASS_ORDER, load_all_datasets
from src.experiments.dl.utils.models import SLM_MODELS, get_model_name

LABEL2ID = {lbl: idx for idx, lbl in enumerate(CLASS_ORDER)}
ID2LABEL = {idx: lbl for idx, lbl in enumerate(CLASS_ORDER)}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Obfuscation Robustness Evaluation")
    p.add_argument("--data_root", type=str, default="data/processed")
    p.add_argument("--out_dir", type=str, default="output/dl/obfuscation")
    p.add_argument("--checkpoints_dir", type=str, default="output/dl/checkpoints")
    p.add_argument("--models", nargs="+", default=list(SLM_MODELS.keys()), help="Model keys to evaluate")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--feature_type", type=str, default="norm")
    return p.parse_args()


def prepare_hf_dataset(texts: list[str], labels: list[str], tokenizer: AutoTokenizer) -> Dataset:
    int_labels = [LABEL2ID[lbl] for lbl in labels]
    ds = Dataset.from_dict({"text": texts, "label": int_labels})
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
    ds = ds.map(tokenize_function, batched=True)
    return ds


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f">> Loading datasets for Obfuscation Analysis...", flush=True)
    data_path = Path(args.data_root) / "metasms_v1_enriched.csv"
    if not data_path.exists():
        data_path = Path(args.data_root) / "metasms_v1_privacy.csv"
        print(f"   (Enriched not found, using fallback: {data_path.name})")
        
    df = pd.read_csv(data_path, dtype=str)
    df = df.dropna(subset=["canonical_label"])
    df = df[df["canonical_label"].isin(CLASS_ORDER)]
    
    splits_all = load_all_datasets(args.data_root, feature_type=args.feature_type)
    metasms = splits_all.get("metasms")
    
    texts = metasms.texts
    labels = metasms.labels
    
    # Align obfuscation tags exactly with the extracted labels
    base_df = metasms.df
    
    # Filter the original dataframe using the exact indices of the filtered base_df
    # This prevents shape mismatches (e.g. ValueError in train_test_split) 
    filtered_df = df.loc[base_df.index]
    
    # Safely extract obfuscation tags
    if "obfuscation_tags" in filtered_df.columns:
        has_obf = (filtered_df["obfuscation_tags"].fillna("").astype(str).str.len() > 0).tolist()
    else:
        print("!! WARNING: 'obfuscation_tags' column not found. All mapped to False.")
        has_obf = [False] * len(filtered_df)
    
    np_labels = np.array(labels)
    # Perform exact split tracking texts, labels AND obfuscation status
    X_train, X_temp, y_train, y_temp, obf_train, obf_temp = train_test_split(
        texts, np_labels, has_obf, test_size=0.2, random_state=42, stratify=np_labels
    )
    X_val, X_test, y_val, y_test, obf_val, obf_test = train_test_split(
        X_temp, y_temp, obf_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    test_df = pd.DataFrame({
        "text": X_test,
        "label": y_test,
        "is_obfuscated": obf_test
    })
    
    obfuscated_test = test_df[test_df["is_obfuscated"] == True]
    
    results_file = out_dir / "results_obfuscation.csv"
    
    completed_models = set()
    if results_file.exists() and results_file.stat().st_size > 0:
        try:
            df_prev = pd.read_csv(results_file)
            for _, row in df_prev.iterrows():
                completed_models.add(row["model_key"])
            print(f">> Found existing results. Skipping {len(completed_models)} completed models.")
        except Exception as e:
            print(f"!! Failed to read existing results file: {e}")
            
    print(f">> Starting Obfuscation evaluation on Test Set (N_Obfuscated={len(obfuscated_test)} out of N_Total={len(test_df)})...", flush=True)
    
    if len(obfuscated_test) == 0:
        print("!! No obfuscated messages found in the test set. Aborting.")
        return

    for model_key in args.models:
        if model_key in completed_models:
            print(f"Skipping {model_key} (Already completed)")
            continue
            
        model_path = Path(args.checkpoints_dir) / model_key
        if not model_path.exists():
            print(f"Skipping {model_key}: No checkpoint found at {model_path}")
            continue
            
        print(f"\n=== Evaluating {model_key} ===", flush=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception:
            hf_id = get_model_name(model_key)
            tokenizer = AutoTokenizer.from_pretrained(hf_id)
            
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        training_args = TrainingArguments(
            output_dir=str(out_dir / "temp"),
            per_device_eval_batch_size=args.batch_size,
            report_to="none",
            disable_tqdm=True
        )
        trainer = Trainer(model=model, args=training_args)
        
        # Evaluate ONLY on obfuscated samples
        ds = prepare_hf_dataset(obfuscated_test["text"].tolist(), obfuscated_test["label"].tolist(), tokenizer)
        preds_output = trainer.predict(ds)
        
        preds = np.argmax(preds_output.predictions, axis=1)
        true_labels = [LABEL2ID[lbl] for lbl in obfuscated_test["label"]]
        
        macro_f1 = float(f1_score(true_labels, preds, average="macro", labels=list(ID2LABEL.keys()), zero_division=0))
        acc = float(accuracy_score(true_labels, preds))
        
        res_dict = {
            "model_key": model_key,
            "n_samples": len(obfuscated_test),
            "macro_f1": macro_f1,
            "accuracy": acc
        }
        
        df_res = pd.DataFrame([res_dict])
        has_headers = results_file.exists() and results_file.stat().st_size > 0
        df_res.to_csv(results_file, mode="a", index=False, header=not has_headers)
        
        # Aggressive memory cleanup
        del model
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print(">> Obfuscation evaluation finished.")

if __name__ == "__main__":
    main()
