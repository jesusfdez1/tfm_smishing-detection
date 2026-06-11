"""
Registry for Small Language Models (SLMs) used in the Deep Learning pipeline.
These are Hugging Face model hub identifiers.
"""

# The 7 chosen models (Spanish + Multilingual + Lightweight)
SLM_MODELS = {
    "beto": "dccuchile/bert-base-spanish-wwm-cased",
    "roberta-es": "PlanTL-GOB-ES/roberta-base-bne",
    "mbert": "bert-base-multilingual-cased",
    "xlm-r": "xlm-roberta-base",
    "mdeberta": "microsoft/mdeberta-v3-base",
    "distilbert": "distilbert-base-multilingual-cased",
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "securebert2": "cisco-ai/SecureBERT2.0-base"
}

def get_model_name(key: str) -> str:
    """Returns the HF model name for a given key, or the key itself if not found."""
    return SLM_MODELS.get(key, key)
