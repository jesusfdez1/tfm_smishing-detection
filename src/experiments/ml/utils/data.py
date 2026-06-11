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

    @property
    def texts(self) -> list[str]:
        return self.df["text_norm"].tolist()

    @property
    def labels(self) -> list[str]:
        return self.df["label"].tolist()

    @property
    def n(self) -> int:
        return len(self.df)

    def class_distribution(self) -> dict[str, int]:
        vc = self.df["label"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
        return {lbl: int(vc.loc[lbl]) for lbl in CLASS_ORDER}


def _attach_norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_norm"] = df["text"].astype(str).map(normalize_text)
    df = df[df["text_norm"].str.len() > 0].copy()
    return df


def load_all_datasets(data_root: str | Path = "data/processed") -> dict[str, Splits]:
    """Loads the 400k SMS dataset (MetaSMS) and prepares it for ML."""
    root = Path(data_root)
    path = root / "metasms_v1_privacy.csv"
    
    if not path.exists():
        path = root / "metasms_v1_enriched.csv"
        
    df = pd.read_csv(path, dtype=str)
    
    # Ensure we have the text column to normalize and the canonical label
    text_col = "text_anonymized" if "text_anonymized" in df.columns else "text"
    df = df.dropna(subset=[text_col, "canonical_label"])
    
    # Filter to keep only valid classes
    df = df[df["canonical_label"].isin(CLASS_ORDER)]
    
    out = pd.DataFrame({
        "text": df[text_col].astype(str),
        "label": df["canonical_label"].astype(str).str.lower().str.strip(),
        "source": "metasms",
    })
    
    out = _attach_norm(out)
    
    return {
        "metasms": Splits("MetaSMS", out)
    }



def summarize(splits: dict[str, Splits]) -> pd.DataFrame:
    """Summary table of available datasets."""
    rows = []
    for key, sp in splits.items():
        dist = sp.class_distribution()
        rows.append({
            "Dataset": sp.name,
            "Messages": sp.n,
            "Ham": dist.get("ham", 0),
            "Spam": dist.get("spam", 0),
            "Smishing": dist.get("smishing", 0),
        })
    return pd.DataFrame(rows).set_index("Dataset")


if __name__ == "__main__":  # quick smoke-test
    splits = load_all_datasets()
    print(summarize(splits))
