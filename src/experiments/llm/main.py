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
from src.experiments.llm.utils.prompts import get_system_prompt, build_zero_shot_prompt, build_few_shot_prompt
from src.experiments.llm.utils.local_clients import LocalLLMClient

# Load environment variables
load_dotenv()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM Evaluation Pipeline")
    p.add_argument("--data_root", type=str, default="data/processed")
    p.add_argument("--out_dir", type=str, default="output/llm")
    p.add_argument("--models", nargs="+", default=["phi4_mini", "qwen25_14b"], help="Model keys to evaluate")
    p.add_argument("--test_samples", type=int, default=1000, help="Number of test samples to evaluate")
    p.add_argument("--smoke_test", action="store_true", help="Run a fast subset for testing")
    return p.parse_args()

LOCAL_MODELS = {
    # Small / Edge Models (< 10B)
    "phi4_mini": "microsoft/Phi-4-Mini-Instruct",
    "gemma3_4b": "google/gemma-3-4b-it",
    "qwen35_4b": "Qwen/Qwen3.5-4B",
    "ministral_8b": "mistralai/Ministral-8B-Instruct-2410",
    "olmoe_1b_7b": "allenai/OLMoE-1B-7B-0924-Instruct", # Modelo MoE para Edge (7B total, 1B activo)
    "falcon3_7b": "tiiuae/Falcon3-7B-Instruct",         # TII (Falcon 3)
    
    # Medium / Server Models (10B - 35B) -> Optimizados para 2 GPUs (80GB VRAM)
    "moonlight_16b": "moonshotai/Moonlight-16B-A3B-Instruct",   # Moonshot AI (Kimi, 2026)
    "gemma4_12b": "google/gemma-4-12B-it",                      # Google (Frontera 2026)
    "deepseek_r1_14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", # DeepSeek (Razonamiento 14B)
    "mistral_nemo_12b": "mistralai/Mistral-Nemo-Instruct-2407",  # Mistral AI (12B)
    "qwen25_14b": "Qwen/Qwen2.5-14B-Instruct",                  # Alibaba Qwen (14B)
    "qwen35_35b_moe": "Qwen/Qwen3.5-35B-A3B"                    # Alibaba MoE masivo (35B)
}

def init_client(model_key: str):
    hf_model_id = LOCAL_MODELS.get(model_key)
    if not hf_model_id:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(LOCAL_MODELS.keys())}")
    
    return LocalLLMClient(model_id=hf_model_id)

import re

def extract_label_from_json(response_text: str) -> str:
    """Safely extracts the label from a JSON response."""
    text = response_text.strip()
    
    # Remove DeepSeek-R1 reasoning blocks if present
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # In case the generation cut off without closing </think>
    if '<think>' in text:
        text = text.split('</think>')[-1] # take everything after closing, or empty
        text = text.replace('<think>', '').strip()
        
    # Clean markdown wrappers if present
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    try:
        data = json.loads(text)
        label = str(data.get("label", "")).lower().strip()
        if label in ["ham", "spam", "smishing"]:
            return label
    except json.JSONDecodeError:
        pass
        
    # Fallback heuristic if JSON is broken
    text_lower = text.lower()
    if "smishing" in text_lower: return "smishing"
    if "spam" in text_lower: return "spam"
    if "ham" in text_lower: return "ham"
    
    return "ham" # Ultimate fallback class

def evaluate_llm(client, texts: list[str], labels: list[str], config: dict, few_shot_examples: list[dict] = None) -> tuple[dict, list, list, list]:
    system_prompt = get_system_prompt(persona=config["persona"], reasoning=config["reasoning"])
    
    print(f">> Preparing {len(texts)} prompts for vLLM...", flush=True)
    raw_prompts = []
    for text in texts:
        if config["context"] == "few-shot" and few_shot_examples:
            raw_prompts.append(build_few_shot_prompt(text, few_shot_examples))
        else:
            raw_prompts.append(build_zero_shot_prompt(text))
            
    t0 = time.perf_counter()
    
    # Generate all responses in parallel using vLLM continuous batching
    try:
        raw_responses = client.generate_batch(raw_prompts, system_prompt=system_prompt)
    except Exception as e:
        print(f"!! Critical vLLM generation error: {e}")
        raw_responses = [f"ERROR: {str(e)}"] * len(texts)

    predict_time = time.perf_counter() - t0
    
    y_pred = []
    y_true = labels
    
    # Parse all generated responses
    for i, response in enumerate(raw_responses):
        try:
            final_pred = extract_label_from_json(response)
            y_pred.append(final_pred)
        except Exception as e:
            print(f"Error parsing sample {i}: {e}")
            y_pred.append("ham")
    
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=CLASS_ORDER, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    
    res_dict = {
        "val_macro_f1": 0.0,
        "val_accuracy": 0.0,
        "test_macro_f1": float(macro_f1),
        "test_accuracy": float(acc),
        "test_f1_ham": float(per_class_f1[0]),
        "test_f1_spam": float(per_class_f1[1]),
        "test_f1_smishing": float(per_class_f1[2]),
        "fit_time_s": 0.0,
        "predict_time_s": predict_time
    }
    
    return res_dict, cm, y_pred, raw_responses

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
    
    target_samples = 10 if args.smoke_test else args.test_samples
    target_samples = min(target_samples, len(X_test))
    
    X_test_sub, _, y_test_sub, _ = train_test_split(
        X_test, y_test, train_size=target_samples, random_state=42, stratify=y_test
    )
    
    print(f">> LLM Evaluation Test Set Sub-sampled to {len(X_test_sub)} samples.", flush=True)
    
    # Deterministic few-shot examples (Cross-lingual: English examples -> Spanish classification)
    # This guarantees prompt reproducibility and evaluates the LLM's cross-lingual abstraction capability.
    few_shot_examples = [
        {"label": "ham", "text": "Hi Maria, we are going to have dinner at 9:00 PM. See you at the door."},
        {"label": "spam", "text": "Exclusive offer! Get 20% off your next order using the code VIP20."},
        {"label": "smishing", "text": "BANK: Your card has been temporarily blocked for security. Verify your details urgently at: http://bit.ly/auth-bank"}
    ]

    # Prompt Grid Search Configurations (Experiments 6 & 7)
    grid_configs = [
        {"name": "zero-shot_basic_direct", "context": "zero-shot", "persona": "basic", "reasoning": "direct"},
        {"name": "zero-shot_expert_direct", "context": "zero-shot", "persona": "expert", "reasoning": "direct"},
        {"name": "zero-shot_expert_cot", "context": "zero-shot", "persona": "expert", "reasoning": "cot"},
        {"name": "few-shot_expert_cot", "context": "few-shot", "persona": "expert", "reasoning": "cot"},
    ]
    
    if args.smoke_test:
        grid_configs = [grid_configs[0], grid_configs[-1]] # Test extreme edges for speed
        
    results_file = out_dir / "results.csv"
    cm_file = out_dir / "confusion_matrices.csv"
    raw_file = out_dir / "raw_predictions.csv"
    
    for model_key in args.models:
        print(f"\n=======================================================", flush=True)
        print(f"=== Evaluating {model_key} ===", flush=True)
        print(f"=======================================================", flush=True)
        
        # Check if all variants for this model are already evaluated before loading into VRAM
        all_evaluated = False
        if results_file.exists():
            df_existing = pd.read_csv(results_file)
            if "model_key" in df_existing.columns:
                evaluated_variants = df_existing["model_key"].values
                all_evaluated = all(
                    f"{model_key}_{config['name']}" in evaluated_variants
                    for config in grid_configs
                )
                
        if all_evaluated:
            print(f">> All variants for {model_key} already evaluated. Skipping model load.", flush=True)
            continue
            
        try:
            client = init_client(model_key)
        except Exception as e:
            print(f"!! ERROR initializing {model_key}: {e}", flush=True)
            continue
            
        for config in grid_configs:
            variant_name = config["name"]
            model_variant = f"{model_key}_{variant_name}"
            
            if results_file.exists():
                df_existing = pd.read_csv(results_file)
                if "model_key" in df_existing.columns and model_variant in df_existing["model_key"].values:
                    print(f">> Skipping {model_variant}, already evaluated.", flush=True)
                    continue
            
            print(f">> Running {variant_name}...", flush=True)
            
            res_dict, cm, y_pred, raw_responses = evaluate_llm(client, X_test_sub, y_test_sub, config, few_shot_examples=few_shot_examples)
            
            res_dict["dataset"] = "MetaSMS"
            res_dict["model_key"] = f"{model_key}_{variant_name}"
            res_dict["n_train"] = 0
            res_dict["n_val"] = 0
            res_dict["n_test"] = len(y_test_sub)
            
            df_res = pd.DataFrame([res_dict])
            df_res.to_csv(results_file, mode="a", index=False, header=not results_file.exists())
            
            cm_records = []
            for i, true_lbl in enumerate(CLASS_ORDER):
                for j, pred_lbl in enumerate(CLASS_ORDER):
                    cm_records.append({
                        "dataset": "MetaSMS",
                        "model_key": f"{model_key}_{variant_name}",
                        "true_label": true_lbl,
                        "predicted_label": pred_lbl,
                        "count": int(cm[i, j])
                    })
            df_cm = pd.DataFrame(cm_records)
            df_cm.to_csv(cm_file, mode="a", index=False, header=not cm_file.exists())
            
            # Save raw responses for qualitative analysis (Chain of Thought, JSON formatting errors, etc)
            df_raw = pd.DataFrame({
                "dataset": "MetaSMS",
                "model_key": f"{model_key}_{variant_name}",
                "true_label": y_test_sub,
                "predicted_label": y_pred,
                "sms_text": X_test_sub,
                "raw_llm_response": raw_responses
            })
            df_raw.to_csv(raw_file, mode="a", index=False, header=not raw_file.exists())
            
            print(f">> OK: {model_key} {variant_name} evaluation saved.", flush=True)

        # Destruir el engine de vLLM y vaciar la VRAM para que el siguiente modelo no lance Out Of Memory
        if 'client' in locals() and client is not None:
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel
                destroy_model_parallel()
            except ImportError:
                pass
            
            if hasattr(client, 'llm'):
                del client.llm
            del client
            
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            print(f">> [CLEANUP] VRAM liberada tras evaluar {model_key}.", flush=True)

    print(">> LLM pipeline finished.", flush=True)

if __name__ == "__main__":
    main()
