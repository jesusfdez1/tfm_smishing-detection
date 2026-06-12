"""Data loading and normalization module for Machine Learning.

Exposes the unified TFM dataset (`MetaSMS`) ready for training.
This dataset contains ~400,000 messages labeled as ham, spam, or smishing.

The schema returned in the Splits objects includes:
    columns = ["text", "text_norm", "label", "source"]
    label  in {"ham", "spam", "smishing"}
    source = "metasms"

`text_norm` is the normalized version used as input for the statistical 
and embedding encoders (BoW, TF-IDF, Word2Vec, FastText).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


CLASS_ORDER = ["ham", "spam", "smishing"]

# Common replacements to uniform the texts.
# We detect these elements and translate them to the SAME flat token format
# (without <>) that sklearn can naturally tokenize.
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\s\-]{6,}\d")
LONG_NUMBER_RE = re.compile(r"\b\d{4,}\b")
ANGLE_TOKEN_RE = re.compile(r"<\s*([A-Z_][A-Z0-9_]*)\s*>")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(raw: str) -> str:
    """Normalizes an SMS message to a string ready for vectorization.

    - Converts to lowercase.
    - Replaces URLs, emails, and phone numbers with stable tokens.
    - Converts anonymous `<URL>`, `<EMAIL_ADDRESS>`, etc. tokens
      to a flat word representation.
    - Collapses repeated spaces.
    """
    if raw is None:
        return ""
    s = str(raw)
    s = ANGLE_TOKEN_RE.sub(lambda m: f" {m.group(1).lower()}_token ", s)
    s = URL_RE.sub(" url_token ", s)
    s = EMAIL_RE.sub(" email_address_token ", s)
    s = PHONE_RE.sub(" phone_number_token ", s)
    s = LONG_NUMBER_RE.sub(" number_token ", s)
    s = s.lower()
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s


@dataclass(frozen=True)
class Splits:
    """Evaluation set ready for hold-out or cross-validation."""

    name: str
    df: pd.DataFrame
    feature_type: str = "norm"

    @property
    def texts(self) -> list[str]:
        target_col = f"text_{self.feature_type}"
        if target_col not in self.df.columns:
            # Fallback to norm if the requested feature is missing
            target_col = "text_norm"
        return self.df[target_col].tolist()

    @property
    def labels(self) -> list[str]:
        return self.df["label"].tolist()

    @property
    def n(self) -> int:
        return len(self.df)

    def class_distribution(self) -> dict[str, int]:
        vc = self.df["label"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
        return {lbl: int(vc.loc[lbl]) for lbl in CLASS_ORDER}


def _attach_norm(df: pd.DataFrame, source_col: str = "text") -> pd.DataFrame:
    df = df.copy()
    df["text_norm"] = df[source_col].astype(str).map(normalize_text)
    # Don't drop here, just let empty strings exist, we drop later globally
    return df


def load_all_datasets(data_root: str | Path = "data/processed", feature_type: str = "norm") -> dict[str, Splits]:
    """Loads the 400k SMS dataset (MetaSMS) and prepares it for ML/DL.
    
    Args:
        data_root: Path to the processed data directory.
        feature_type: Which text representation to use ('norm', 'raw', 'injected').
    """
    root = Path(data_root)
    # Prefer the enriched dataset to support injected features, fallback to privacy
    path = root / "metasms_v1_enriched.csv"
    if not path.exists():
        path = root / "metasms_v1_privacy.csv"
        
    df = pd.read_csv(path, dtype=str)
    
    # Ensure we have the canonical label
    df = df.dropna(subset=["canonical_label"])
    df = df[df["canonical_label"].isin(CLASS_ORDER)]
    
    # Extract core columns safely
    out = pd.DataFrame({
        "label": df["canonical_label"].astype(str).str.lower().str.strip(),
        "source": "metasms",
    })
    
    # 1. Feature: Raw (un-anonymized)
    if "text_raw" in df.columns:
        out["text_raw"] = df["text_raw"].astype(str).fillna("")
    else:
        out["text_raw"] = df["text"].astype(str).fillna("") # Fallback
        
    # 2. Feature: Anonymized
    if "text_anonymized" in df.columns:
        out["text_anonymized"] = df["text_anonymized"].astype(str).fillna("")
    else:
        out["text_anonymized"] = out["text_raw"]
        
    # 3. Feature: Norm (normalized anonymized)
    out = _attach_norm(out, source_col="text_anonymized")
    
    # 4. Feature: Injected (LLM metadata + normalized text)
    if "urgency_level" in df.columns and "theme" in df.columns:
        # Create injected representation: "[URGENCIA: Alta] [TEMA: Bancario] texto..."
        urgency = df["urgency_level"].fillna("Desconocida").astype(str)
        theme = df["theme"].fillna("General").astype(str)
        
        injected_texts = []
        for urg, thm, txt in zip(urgency, theme, out["text_norm"]):
            prefix = f"[URGENCIA: {urg}] [TEMA: {thm}]"
            injected_texts.append(f"{prefix} {txt}")
        out["text_injected"] = injected_texts
    else:
        # Fallback if metadata is missing
        out["text_injected"] = out["text_norm"]

    # Filter out empty messages
    target_col = f"text_{feature_type}"
    if target_col not in out.columns:
        target_col = "text_norm"
    out = out[out[target_col].str.len() > 0].copy()
    
    return {
        "metasms": Splits("MetaSMS", out, feature_type=feature_type)
    }



def summarize(splits: dict[str, Splits]) -> pd.DataFrame:
    """Summary table of available datasets."""
    rows = []
    for key, sp in splits.items():
        dist = sp.class_distribution()
        rows.append({
            "Dataset": sp.name,
            "Feature View": sp.feature_type,
            "Messages": sp.n,
            "Ham": dist.get("ham", 0),
            "Spam": dist.get("spam", 0),
            "Smishing": dist.get("smishing", 0),
        })
    return pd.DataFrame(rows).set_index("Dataset")


if __name__ == "__main__":  # quick smoke-test
    print("Testing 'norm' view:")
    splits_norm = load_all_datasets(feature_type="norm")
    print(summarize(splits_norm))
    
    print("\nTesting 'injected' view:")
    splits_inj = load_all_datasets(feature_type="injected")
    print(summarize(splits_inj))
