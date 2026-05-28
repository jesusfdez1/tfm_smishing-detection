import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Mapeo estricto de las 3 clases de nuestro TFM
LABEL_MAP = {"ham": 0, "spam": 1, "smishing": 2}
REVERSE_LABEL_MAP = {0: "ham", 1: "spam", 2: "smishing"}

def load_and_prepare_data(csv_path: str, text_col: str) -> Tuple[pd.Series, pd.Series]:
    """Carga el dataset maestro y prepara las columnas X e y."""
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)
    
    # Asegurar que no hay nulos en el texto ni en las etiquetas
    df = df.dropna(subset=[text_col, "canonical_label"])
    
    # Filtrar solo las 3 clases oficiales (por si acaso hubiera ruido)
    df = df[df["canonical_label"].isin(LABEL_MAP.keys())]
    
    X = df[text_col].astype(str)
    y = df["canonical_label"].map(LABEL_MAP)
    
    logger.info(f"Loaded {len(X)} records for training.")
    return X, y

def train_sklearn_model(X_train, y_train, model_type: str):
    """Entrena un modelo clásico usando TF-IDF."""
    logger.info("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    
    if model_type == "logistic":
        logger.info("Training Logistic Regression baseline...")
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    elif model_type == "random_forest":
        logger.info("Training Random Forest...")
        model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown sklearn model type: {model_type}")
        
    model.fit(X_train_vec, y_train)
    return model, vectorizer

def evaluate_and_save(model, vectorizer, X_test, y_test, output_dir: Path, exp_name: str):
    """Evalúa el modelo y guarda las métricas y los pesos."""
    logger.info("Evaluating model on test set...")
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    
    target_names = ["ham", "spam", "smishing"]
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    # Mostrar resultados por pantalla
    print("\n" + "="*50)
    print(f"RESULTS FOR EXPERIMENT: {exp_name}")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Guardar métricas en JSON
    metrics_path = output_dir / "logs" / f"{exp_name}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "experiment_name": exp_name,
        "classification_report": report,
        "confusion_matrix": cm
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Guardar pesos del modelo
    model_path = output_dir / "models" / f"{exp_name}_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer}, f)
    logger.info(f"Saved model weights to {model_path}")

def main():
    parser = argparse.ArgumentParser(description="Script maestro de entrenamiento para el TFM")
    parser.add_argument("--data", type=str, required=True, help="Ruta al dataset CSV")
    parser.add_argument("--text-col", type=str, choices=["text", "text_anonymized"], required=True, 
                        help="¿Usar texto original (text) o anonimizado (text_anonymized)?")
    parser.add_argument("--model", type=str, choices=["logistic", "random_forest", "deberta"], default="logistic",
                        help="Modelo a entrenar. Nota: deberta requerirá implementación en PyTorch/HuggingFace")
    parser.add_argument("--exp-name", type=str, required=True, help="Nombre único del experimento (ej: baseline_raw_text)")
    parser.add_argument("--output-dir", type=str, default="experiments/results", help="Carpeta base para guardar resultados")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    X, y = load_and_prepare_data(args.data, args.text_col)
    
    # Dividir estratificado para mantener proporción ham/spam/smishing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if args.model in ["logistic", "random_forest"]:
        model, vectorizer = train_sklearn_model(X_train, y_train, args.model)
        evaluate_and_save(model, vectorizer, X_test, y_test, output_dir, args.exp_name)
    elif args.model == "deberta":
        logger.error("DeBERTa implementation goes here! Use HuggingFace Trainer API.")
        # TODO: Implementar AutoModelForSequenceClassification para el TFM.
    
if __name__ == "__main__":
    main()
