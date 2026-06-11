"""Entry point for the LLM Zero-Shot/Few-Shot evaluation pipeline.

Evaluates GPT, Gemini, and DeepSeek on a subset of the test dataset.

Usage:
    python -m src.experiments.llm.main --models gpt gemini deepseek
    python -m src.experiments.llm.main --smoke_test
"""

import argparse
import json
import time
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from dotenv import load_dotenv

from src.experiments.ml.utils.data import load_all_datasets, CLASS_ORDER
from src.experiments.llm.utils.prompts import SYSTEM_PROMPT, build_zero_shot_prompt, build_few_shot_prompt
from src.experiments.llm.utils.api_clients import OpenAIClient, GeminiClient, DeepSeekClient

# Load environment variables
load_dotenv()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM Evaluation Pipeline")
    p.add_argument("--data_root", type=str, default="data/processed")
    p.add_argument("--out_dir", type=str, default="output/llm")
    p.add_argument("--models", nargs="+", default=["gpt", "gemini", "deepseek"], help="Models to evaluate")
    p.add_argument("--test_samples", type=int, default=1000, help="Number of test samples to evaluate")
    p.add_argument("--smoke_test", action="store_true", help="Run a fast subset for testing")
    return p.parse_args()

def init_client(model_key: str):
    if model_key == "gpt":
        return OpenAIClient(model_name="gpt-4o-mini")
    elif model_key == "gemini":
        return GeminiClient(model_name="gemini-2.5-flash")
    elif model_key == "deepseek":
        return DeepSeekClient(model_name="deepseek-chat")
    else:
        raise ValueError(f"Unknown model: {model_key}")

def evaluate_llm(client, texts: list[str], labels: list[str], few_shot_examples: list[dict] = None) -> tuple[dict, list, list]:
    y_pred = []
    y_true = []
    
    t0 = time.perf_counter()
    for i, (text, label) in enumerate(zip(texts, labels)):
        if few_shot_examples:
            prompt = build_few_shot_prompt(text, few_shot_examples)
        else:
            prompt = build_zero_shot_prompt(text)
            
        try:
            prediction = client.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
            
            # Post-process output to strictly match CLASS_ORDER
            pred_clean = prediction.strip().lower()
            if "smishing" in pred_clean:
                final_pred = "smishing"
            elif "spam" in pred_clean:
                final_pred = "spam"
            elif "ham" in pred_clean:
                final_pred = "ham"
            else:
                final_pred = "ham" # Fallback
                
            y_pred.append(final_pred)
            y_true.append(label)
        except Exception as e:
            print(f"Error evaluating sample {i}: {e}")
            # Append fallback to keep lists aligned
            y_pred.append("ham")
            y_true.append(label)
            
        # Optional: Print progress every 100 samples
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(texts)} samples...", flush=True)

    predict_time = time.perf_counter() - t0
    
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=CLASS_ORDER, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    
    res_dict = {
        "val_macro_f1": 0.0, # Not applicable
        "val_accuracy": 0.0, # Not applicable
        "test_macro_f1": float(macro_f1),
        "test_accuracy": float(acc),
        "test_f1_ham": float(per_class_f1[0]),
        "test_f1_spam": float(per_class_f1[1]),
        "test_f1_smishing": float(per_class_f1[2]),
        "fit_time_s": 0.0, # Not applicable for inference-only
        "predict_time_s": predict_time
    }
    
    return res_dict, cm, y_pred

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(">> Loading datasets for LLM evaluation...", flush=True)
    splits_all = load_all_datasets(args.data_root)
    metasms = splits_all.get("metasms")
    if not metasms:
        raise ValueError("MetaSMS dataset not found!")

    texts = metasms.texts
    labels = metasms.labels
    
    # 1. Reproduce exact same 80/10/10 split
    import numpy as np
    np_labels = np.array(labels)
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, np_labels, test_size=0.2, random_state=42, stratify=np_labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    y_train = y_train.tolist()
    y_test = y_test.tolist()
    
    # 2. Sub-sample test set (e.g. 1000 items) to save costs
    target_samples = 10 if args.smoke_test else args.test_samples
    
    # Ensure we don't request more samples than available
    target_samples = min(target_samples, len(X_test))
    
    X_test_sub, _, y_test_sub, _ = train_test_split(
        X_test, y_test, train_size=target_samples, random_state=42, stratify=y_test
    )
    
    print(f">> LLM Evaluation Test Set Sub-sampled to {len(X_test_sub)} samples.", flush=True)
    
    # 3. Create Few-Shot examples from Train set (1 of each class)
    few_shot_examples = []
    added_classes = set()
    for txt, lbl in zip(X_train, y_train):
        if lbl not in added_classes:
            few_shot_examples.append({"text": txt, "label": lbl})
            added_classes.add(lbl)
        if len(added_classes) == 3:
            break
            
    modes = ["zero-shot", "few-shot"]
    if args.smoke_test:
        modes = ["zero-shot"] # Keep it fast for smoke test
        
    results_file = out_dir / "results.csv"
    cm_file = out_dir / "confusion_matrices.csv"
    
    for model_key in args.models:
        print(f"\n=======================================================", flush=True)
        print(f"=== Evaluating {model_key} ===", flush=True)
        print(f"=======================================================", flush=True)
        
        try:
            client = init_client(model_key)
        except Exception as e:
            print(f"!! ERROR initializing {model_key}: {e}", flush=True)
            continue
            
        for mode in modes:
            print(f">> Running {mode}...", flush=True)
            
            examples = few_shot_examples if mode == "few-shot" else None
            
            res_dict, cm, _ = evaluate_llm(client, X_test_sub, y_test_sub, few_shot_examples=examples)
            
            res_dict["dataset"] = "MetaSMS"
            res_dict["model_key"] = f"{model_key}_{mode}"
            res_dict["n_train"] = 0
            res_dict["n_val"] = 0
            res_dict["n_test"] = len(y_test_sub)
            
            # Save metrics
            df_res = pd.DataFrame([res_dict])
            df_res.to_csv(results_file, mode="a", index=False, header=not results_file.exists())
            
            # Save CM
            cm_records = []
            for i, true_lbl in enumerate(CLASS_ORDER):
                for j, pred_lbl in enumerate(CLASS_ORDER):
                    cm_records.append({
                        "dataset": "MetaSMS",
                        "model_key": f"{model_key}_{mode}",
                        "true_label": true_lbl,
                        "predicted_label": pred_lbl,
                        "count": int(cm[i, j])
                    })
            df_cm = pd.DataFrame(cm_records)
            df_cm.to_csv(cm_file, mode="a", index=False, header=not cm_file.exists())
            
            print(f">> OK: {model_key} {mode} evaluation saved.", flush=True)

    print(">> LLM pipeline finished.", flush=True)

if __name__ == "__main__":
    main()
