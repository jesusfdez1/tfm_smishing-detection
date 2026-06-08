import argparse
import json
import logging
import pickle
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LABEL_MAP = {"ham": 0, "spam": 1, "smishing": 2}
REVERSE_LABEL_MAP = {0: "ham", 1: "spam", 2: "smishing"}

def plot_confusion_matrix(cm, labels, output_path):
    """Generate and save a heatmap of the confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel('True Label')
    plt.xlabel('Model Prediction')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained model (project)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model's .pkl file")
    parser.add_argument("--test-data", type=str, required=True, help="Evaluation CSV dataset")
    parser.add_argument("--text-col", type=str, choices=["text", "text_anonymized"], required=True)
    parser.add_argument("--output-dir", type=str, default="experiments/results/logs")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading model from {args.model_path}")
    with open(args.model_path, "rb") as f:
        data = pickle.load(f)
        model = data["model"]
        vectorizer = data.get("vectorizer")
        
    logger.info(f"Loading test data from {args.test_data}")
    df = pd.read_csv(args.test_data, dtype=str).dropna(subset=[args.text_col, "canonical_label"])
    df = df[df["canonical_label"].isin(LABEL_MAP.keys())]
    
    X_test = df[args.text_col].astype(str)
    y_test = df["canonical_label"].map(LABEL_MAP)
    
    if vectorizer:
        X_test_vec = vectorizer.transform(X_test)
    else:
        # TODO: Logic to load HuggingFace tokenizers if the model is DeBERTa
        raise NotImplementedError("Only TF-IDF models are fully implemented in this baseline script.")
        
    logger.info("Predicting...")
    y_pred = model.predict(X_test_vec)
    
    target_names = ["ham", "spam", "smishing"]
    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*50)
    print(f"EVALUATION REPORT")
    print("="*50)
    print(report)
    
    model_name = Path(args.model_path).stem
    
    # Save text report
    report_path = output_dir / f"{model_name}_eval_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    # Save plot
    plot_path = output_dir / f"{model_name}_confusion_matrix.png"
    plot_confusion_matrix(cm, target_names, plot_path)
    
if __name__ == "__main__":
    main()
