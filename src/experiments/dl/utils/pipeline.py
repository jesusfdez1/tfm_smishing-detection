"""
Deep Learning (SLM) pipeline implementation.
Handles HF tokenization, datasets conversion, and Trainer orchestration.
"""
from __future__ import annotations

import time
import numpy as np
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import torch

from src.experiments.ml.utils.data import CLASS_ORDER

# Define global label-to-id mapping to ensure consistency
LABEL2ID = {lbl: idx for idx, lbl in enumerate(CLASS_ORDER)}
ID2LABEL = {idx: lbl for idx, lbl in enumerate(CLASS_ORDER)}


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """Computes Macro F1 and Accuracy for HF Trainer."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    macro_f1 = float(f1_score(labels, preds, average="macro", labels=list(ID2LABEL.keys()), zero_division=0))
    acc = float(accuracy_score(labels, preds))
    
    return {
        "macro_f1": macro_f1,
        "accuracy": acc
    }


def prepare_hf_dataset(texts: list[str], labels: list[str], tokenizer: AutoTokenizer) -> Dataset:
    """Converts raw texts and string labels to a tokenized HF Dataset."""
    # Convert string labels to integers based on CLASS_ORDER
    int_labels = [LABEL2ID[lbl] for lbl in labels]
    
    # Create dataset
    ds = Dataset.from_dict({"text": texts, "label": int_labels})
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
    ds = ds.map(tokenize_function, batched=True)
    return ds


def run_slm_evaluation(
    model_name: str,
    hf_model_id: str,
    texts: list[str],
    labels: list[str],
    seed: int = 42,
    batch_size: int = 16,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    out_dir: str = "output/dl/checkpoints",
    smoke_test: bool = False
) -> dict[str, Any]:
    """
    Executes the training and evaluation of a single SLM.
    Uses the strict 80/10/10 stratified split to match the ML pipeline.
    """
    if smoke_test:
        # Reduce dataset aggressively for testing
        print(f"!!! SMOKE TEST MODE FOR {model_name} !!!", flush=True)
        texts = texts[:500]
        labels = labels[:500]
        epochs = 1
        
    # Split 80/10/10 (stratified) exactly like ML
    np_labels = np.array(labels)
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, np_labels, test_size=0.2, random_state=seed, stratify=np_labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    
    y_train = y_train.tolist()
    y_val = y_val.tolist()
    y_test = y_test.tolist()
    
    print(f"[{model_name}] Partitions -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}", flush=True)

    # Initialize Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_id,
        num_labels=len(CLASS_ORDER),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # Prepare datasets
    train_ds = prepare_hf_dataset(X_train, y_train, tokenizer)
    val_ds = prepare_hf_dataset(X_val, y_val, tokenizer)
    test_ds = prepare_hf_dataset(X_test, y_test, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{out_dir}/{model_name}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_dir=f"{out_dir}/logs",
        logging_steps=50,
        report_to="none" # Disable wandb/tensorboard to keep it simple
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    from transformers.trainer_utils import get_last_checkpoint
    import os
    
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            print(f"[{model_name}] Resumiendo entrenamiento desde checkpoint: {last_checkpoint}", flush=True)

    # Train
    t0 = time.perf_counter()
    if last_checkpoint is not None:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
    fit_time = time.perf_counter() - t0

    # Evaluate on Validation (just to record final val metrics)
    val_results = trainer.evaluate(eval_dataset=val_ds)

    # Evaluate on Test
    t1 = time.perf_counter()
    test_predictions = trainer.predict(test_dataset=test_ds)
    predict_time = time.perf_counter() - t1
    
    test_metrics = test_predictions.metrics
    preds = np.argmax(test_predictions.predictions, axis=1)
    
    # Calculate per-class F1 and Confusion Matrix manually on test
    true_labels = [LABEL2ID[lbl] for lbl in y_test]
    per_class_f1 = f1_score(true_labels, preds, average=None, labels=list(ID2LABEL.keys()), zero_division=0)
    cm = confusion_matrix(true_labels, preds, labels=list(ID2LABEL.keys()))

    return {
        "dataset": "MetaSMS",
        "model_key": model_name,
        "hf_id": hf_model_id,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "val_macro_f1": val_results.get("eval_macro_f1", 0.0),
        "val_accuracy": val_results.get("eval_accuracy", 0.0),
        "test_macro_f1": test_metrics.get("test_macro_f1", 0.0),
        "test_accuracy": test_metrics.get("test_accuracy", 0.0),
        "test_f1_ham": float(per_class_f1[0]),
        "test_f1_spam": float(per_class_f1[1]),
        "test_f1_smishing": float(per_class_f1[2]),
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
        "confusion_matrix": cm
    }
