"""Build the MetaSMS-HSS master dataset from heterogeneous raw sources.

This script handles schema normalization, anonymization, language detection,
LLM enrichment, deduplication, and resumable execution.
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Iterable, Optional, Tuple, List

import pandas as pd

from utils.cleaner import DatasetCleaner
from utils.anonymizer import SMSAnonymizer
from utils.date_parser import normalize_timestamp

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Output schema for the master dataset.
MASTER_FIELDS = [
    # 1. Core Data (Target & Features)
    "message_id",
    "canonical_label",
    "text_anonymized",
    "text",
    
    # 2. Provenance (Origin & Traceability)
    "reference",
    "source_id",
    "original_label",
    "label_mapping_rule",
    "timestamp_original",
    "license",
    
    # 3. Processing Metadata
    "language",
    "language_confidence",
    "anonymization_status",
    
    # 4. LLM Enrichment (Phase 2)
    "llm_model",
    "prompt_version",
    "llm_annotation_date",
    "theme",
    "urgency_level",
    "whois_domain",
    "whois_tld",
    "whois_age_days",
    "whois_hidden",
    "whois_country",
]


SOURCE_META = {
    "mishra_soni_2022": {
        "license": "Research Use",
        "source_url": "https://doi.org/10.17632/f45bkkt8pr.1",
    },
    "mishra_extended": {
        "license": "Research Use",
        "source_url": "https://doi.org/10.17632/f45bkkt8pr.1",
    },
    "hosseinpour_2025": {
        "license": "Research Use",
        "source_url": "https://doi.org/10.1145/3734477.3736147",
    },
    "agarwal_2025": {
        "license": "Research Use",
        "source_url": "https://doi.org/10.1145/3730567.3764431",
    },
    "uci_sms_spam": {
        "license": "Research Use",
        "source_url": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
    },
    "nus_sms": {
        "license": "Research Use",
        "source_url": "https://doi.org/10.1007/s10579-012-9197-9",
    },
    "kaggle_spam_ham": {
        "license": "Unknown",
        "source_url": None,
    },
    "kaggle_phishing": {
        "license": "Unknown",
        "source_url": None,
    },

    "exais_sms": {
        "license": "Research Use",
        "source_url": None,
    },
    "spanish_spam_ham": {
        "license": "Unknown",
        "source_url": None,
    },
    "malicious_benign_sms_mms": {
        "license": "CC BY-NC 4.0",
        "source_url": None,
    },
    "smishing_4c": {
        "license": "Research Use",
        "source_url": "https://www.kaggle.com/datasets/galactus007/sms-smishing-collection-data-set",
    },
    "mimics_3500": {
        "license": "Research Use",
        "source_url": "https://data.mendeley.com/datasets/f45bkkt8pr/1",
    },
}


def clean_str(value: Any) -> Optional[str]:
    """Clean string by removing surrounding whitespace and internal newlines.
    
    Safely coerces numeric types (e.g., ints from JSON loaders) to strings.
    """
    if value is None or value == "":
        return None
    
    import math
    if isinstance(value, float) and math.isnan(value):
        return None
        
    value = str(value)
    # Replace carriage returns and newlines with a single space
    cleaned = value.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    # Collapse multiple spaces into one
    cleaned = " ".join(cleaned.split())
    return cleaned if cleaned else None


def normalize_language(value: Optional[str]) -> Optional[str]:
    """Resolve a language name or tag to an ISO-639-1 (or -3) code.

    Uses the ``langcodes`` library so any language name recognised by the
    IANA/BCP-47 registry resolves automatically — no hardcoded lookup table
    needed.  2-character codes are returned as-is (lowercased).  Unrecognised
    strings (e.g. 'Mixed (Dutch/French)', 'unknown') return None.
    """
    if not value:
        return None
    raw = str(value).strip()
    # Already a valid 2-char code — pass through.
    if len(raw) == 2:
        return raw.lower()
    try:
        import langcodes
        tag = langcodes.find(raw)
        # .language gives the primary language subtag (ISO 639-1 when available,
        # otherwise ISO 639-3, e.g. 'fil' for Filipino).
        return tag.language
    except (LookupError, Exception):
        return None


def map_label_basic(value: Optional[str]) -> Optional[str]:
    """Map common raw labels to the canonical ham/spam/smishing classes."""
    if not value:
        return None
    raw = str(value).strip().lower()
    if raw in {"ham", "legitimate", "benign"}:
        return "ham"
    if raw in {"spam", "promotional", "marketing"}:
        return "spam"
    if raw in {"smishing", "phishing", "malicious", "fraud"}:
        return "smishing"
    return None


def map_binary_label(value: Any, positive_label: str) -> Optional[str]:
    """Map binary labels (0/1 or yes/no) to a target class or ham."""
    if value is None:
        return None
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes"}:
        return positive_label
    if raw in {"0", "false", "no"}:
        return "ham"
    return None


def iso_from_epoch_ms(value: Any) -> Optional[str]:
    """Convert epoch milliseconds to an ISO timestamp string."""
    if value is None:
        return None
    try:
        ts = int(value)
        return datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def iter_csv_dicts(path: Path, chunk_size: int, **kwargs: Any) -> Iterable[Dict[str, Any]]:
    """Yield dict rows from a CSV using chunked reads for large files.

    Callers may override any pandas read_csv keyword via **kwargs, including
    ``encoding`` (default: utf-8) and ``encoding_errors`` (default: replace).
    """
    # Build defaults that callers can override via kwargs
    read_kwargs: Dict[str, Any] = {
        "encoding": "utf-8",
        "encoding_errors": "replace",
        "on_bad_lines": "skip",
    }
    read_kwargs.update(kwargs)
    for chunk in pd.read_csv(
        path,
        chunksize=chunk_size,
        dtype=str,
        keep_default_na=False,
        **read_kwargs,
    ):
        for record in chunk.to_dict(orient="records"):
            yield record



def detect_pre_anonymized(text: str) -> bool:
    """Detect placeholder tokens that indicate prior anonymization."""
    if not text:
        return False
    return bool(re.search(r"<[A-Z0-9_]+>", text))


def build_dedupe_key(text: str) -> str:
    """Hash a normalized text string for cross-source deduplication.

    The input should be the cleaned+anonymized text **without** OBF annotation
    tags (e.g. <OBF_ZWSP>), so that the same message appearing in two datasets
    — one with obfuscation characters and one without — produces the same hash.
    """
    if not text:
        return ""
    # Strip OBF tags appended by DatasetCleaner before hashing.
    clean = re.sub(r"\s*<OBF_[A-Z_]+>", "", text)
    normalized = unicodedata.normalize("NFKC", clean)
    normalized = " ".join(normalized.lower().split())
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def sanitize_token(value: str) -> str:
    """Make a filename-safe token from arbitrary user input."""
    safe = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip())
    safe = safe.strip("-").lower()
    return safe or "na"


def build_output_name(
    output_dir: Path,
    use_privacy_filter: bool,
    dedupe: bool,
    exclude_ai_generated: bool,
) -> Path:
    """Build a concise output filename based on pipeline options."""
    parts = ["metasms_v1"]
    if not dedupe:
        parts.append("nodedupe")
    if use_privacy_filter:
        parts.append("privacy")
    if not exclude_ai_generated:
        parts.append("with_ai")
    
    return output_dir / ("_".join(parts) + ".csv")


def state_signature(
    raw_dir: Path,
    use_privacy_filter: bool,
    privacy_filter_model: str,
    dedupe: bool,
    chunk_size: int,
    exclude_ai_generated: bool,
) -> Dict[str, Any]:
    """Return a stable signature to validate resume compatibility."""
    return {
        "raw_dir": str(raw_dir),
        "use_privacy_filter": use_privacy_filter,
        "privacy_filter_model": privacy_filter_model,
        "dedupe": dedupe,
        "chunk_size": chunk_size,
        "exclude_ai_generated": exclude_ai_generated,
    }


def load_resume_state(state_path: Path) -> Dict[str, Any]:
    """Load resume state from disk, returning an empty dict if missing."""
    if not state_path.exists():
        return {}
    try:
        with state_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def save_resume_state(state_path: Path, state: Dict[str, Any]) -> None:
    """Persist resume state using a temp file for atomic writes."""
    tmp_path = state_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=True, indent=2)
    
    import time
    for attempt in range(5):
        try:
            tmp_path.replace(state_path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.5)


def detect_language(text: str, hint: Optional[str]) -> Dict[str, Any]:
    """Detect language using local libraries with a hint fallback."""
    if not text or len(text.strip()) < 4:
        if hint:
            return {"language": hint, "language_confidence": 1.0}
        return {"language": "unknown", "language_confidence": 0.0}

    detected = None
    confidence = 0.0

    try:
        import pycld3

        result = pycld3.get_language(text)
        if result and result.language:
            detected = result.language[:2].lower()
            confidence = float(result.probability)
    except Exception:
        pass

    if not detected:
        try:
            import langid

            detected, score = langid.classify(text)
            detected = detected[:2].lower()
            # langid score is not a true probability, but we normalize it loosely or just use 1.0
            confidence = 1.0
        except Exception:
            pass

    if not detected:
        try:
            from langdetect import detect_langs, DetectorFactory
            DetectorFactory.seed = 0  # Make deterministic
            
            candidates = detect_langs(text)
            if candidates:
                detected = candidates[0].lang[:2].lower()
                confidence = float(candidates[0].prob)
        except Exception:
            pass

    if not detected:
        if hint:
            return {"language": hint, "language_confidence": 1.0}
        return {"language": "unknown", "language_confidence": 0.0}

    # If the text is short or detection confidence is low, trust the dataset hint
    if hint and (confidence < 0.95 or len(text.strip()) < 60):
        detected = hint
        confidence = 1.0

    return {"language": detected, "language_confidence": confidence}


def build_master_rows(
    raws: List[Dict[str, Any]],
    cleaner: DatasetCleaner,
    anonymizer: SMSAnonymizer,
) -> List[Optional[Tuple[Dict[str, Any], str]]]:
    """Build normalized master rows and return dedupe keys for a batch of raw inputs."""
    results = [None] * len(raws)
    valid_indices = []
    texts_to_clean = []

    # 1. Extract texts
    for i, raw in enumerate(raws):
        text = clean_str(raw.get("text"))
        if text:
            valid_indices.append(i)
            texts_to_clean.append(text)

    if not texts_to_clean:
        return results

    # 2. Clean texts
    cleaned_texts = []
    for text in texts_to_clean:
        cleaner_result = cleaner.clean(text)
        cleaned_texts.append(cleaner_result.get("cleaned", text))

    # 3. Batched anonymization
    anon_results = anonymizer.process_messages(cleaned_texts)

    # 4. Build final rows
    for i, idx in enumerate(valid_indices):
        raw = raws[idx]
        text = texts_to_clean[i]
        cleaned_text = cleaned_texts[i]
        anon_result = anon_results[i]

        llm_text = anon_result.get("anonymized", cleaned_text)

        if detect_pre_anonymized(text):
            anonymization_status = "pre_anonymized"
        elif llm_text != text:
            anonymization_status = "anonymized"
        else:
            anonymization_status = "raw"

        canonical_label = raw.get("canonical_label")

        language_data = detect_language(llm_text, raw.get("language_hint"))
        language = language_data.get("language", "unknown")
        language_confidence = language_data.get("language_confidence", 0.0)

        dedupe_key = build_dedupe_key(cleaned_text)

        original_label = raw.get("original_label") or "unlabeled"

        loader_rule = raw.get("label_mapping_rule") or ""
        if loader_rule:
            label_mapping_rule = loader_rule
        elif canonical_label:
            label_mapping_rule = f"{original_label}->{canonical_label}"
        else:
            label_mapping_rule = ""

        row = {
            "message_id": f"msg_{uuid.uuid4().hex[:12]}",
            "text": text,
            "text_anonymized": llm_text,
            "anonymization_status": anonymization_status,
            "canonical_label": canonical_label or "",
            "timestamp_original": normalize_timestamp(raw.get("timestamp_original")),
            "reference": raw.get("source") or "unknown",
            "source_id": raw.get("source_id") or "",
            "original_label": original_label,
            "label_mapping_rule": label_mapping_rule,
            "license": raw.get("license") or "",
            "language": language,
            "language_confidence": f"{language_confidence:.2f}" if language_confidence else "",
            "llm_model": "",
            "prompt_version": "",
            "llm_annotation_date": "",
            "theme": "",
            "urgency_level": "",
            "whois_domain": "",
            "whois_tld": "",
            "whois_age_days": "",
            "whois_hidden": "",
            "whois_country": "",
        }
        results[idx] = (row, dedupe_key)

    return results


def build_master_row(
    raw: Dict[str, Any],
    cleaner: DatasetCleaner,
    anonymizer: SMSAnonymizer,
) -> Optional[Tuple[Dict[str, Any], str]]:
    """Convenience wrapper for single row."""
    return build_master_rows([raw], cleaner, anonymizer)[0]


def load_mishra(path: Path, source_name: str, chunk_size: int) -> Iterable[Dict[str, Any]]:
    """Load Mishra/Soni sources with LABEL/TEXT columns."""
    for i, row in enumerate(iter_csv_dicts(path, chunk_size=chunk_size), 1):
        label = clean_str(row.get("LABEL") or row.get("label"))
        text = clean_str(row.get("TEXT") or row.get("text"))
        canonical = map_label_basic(label)
        if not canonical or not text:
            continue
        yield {
            "text": text,
            "canonical_label": canonical,
            "timestamp_original": None,
            "source": source_name,
            "source_id": str(i),
            "source_url": SOURCE_META[source_name]["source_url"],
            "original_label": label,
            "license": SOURCE_META[source_name]["license"],
            "language_hint": "en",
        }


def load_hosseinpour(path: Path, chunk_size: int) -> Iterable[Dict[str, Any]]:
    """Load Hosseinpour dataset with spam and smishing binary labels.

    Label logic:
    - smishing_label == '1'  → smishing  (takes priority)
    - spam_label == '1'      → spam
    - spam_label == '0'      → ham  (explicit negative label)
    - spam_label == '' or 'Smishing' with smishing_label != '1'
                             → None (unlabeled; left for LLM enrichment)
    """
    for i, row in enumerate(iter_csv_dicts(path, chunk_size=chunk_size), 1):
        text = clean_str(row.get("message"))
        spam_label = clean_str(row.get("spam label") or row.get("spam_label")) or ""
        smish_label = clean_str(row.get("smishing label") or row.get("smishing_label")) or ""
        if not text:
            continue
        canonical = None
        rule_applied = "else->unlabeled"
        
        if smish_label.strip() == "1":
            canonical = "smishing"
            rule_applied = "smishing_label==1->smishing"
        elif spam_label.strip() == "1":
            canonical = "spam"
            rule_applied = "spam_label==1->spam"
        elif spam_label.strip() == "0":
            # Explicit negative — both spam and smishing are 0 → ham
            canonical = "ham"
            rule_applied = "spam_label==0->ham"
            
        yield {
            "text": text,
            "canonical_label": canonical,
            "timestamp_original": None,
            "source": "hosseinpour_2025",
            "source_id": str(i),
            "source_url": SOURCE_META["hosseinpour_2025"]["source_url"],
            "original_label": f"spam_label={spam_label};smishing_label={smish_label}",
            "label_mapping_rule": rule_applied,
            "license": SOURCE_META["hosseinpour_2025"]["license"],
            "language_hint": None,
        }


def load_kaggle_spam_ham(path: Path, chunk_size: int) -> Iterable[Dict[str, Any]]:
    """Load Kaggle spam/ham dataset with target/text columns."""
    for i, row in enumerate(iter_csv_dicts(path, chunk_size=chunk_size), 1):
        label = clean_str(row.get("target"))
        text = clean_str(row.get("text"))
        canonical = map_label_basic(label)
        if not canonical or not text:
            continue
        yield {
            "text": text,
            "canonical_label": canonical,
            "timestamp_original": None,
            "source": "kaggle_spam_ham",
            "source_id": str(i),
            "source_url": SOURCE_META["kaggle_spam_ham"]["source_url"],
            "original_label": label,
            "license": SOURCE_META["kaggle_spam_ham"]["license"],
            "language_hint": None,
        }


def load_kaggle_phishing(path: Path, chunk_size: int) -> Iterable[Dict[str, Any]]:
    """Load Kaggle phishing dataset and map to smishing."""
    for i, row in enumerate(iter_csv_dicts(path, chunk_size=chunk_size), 1):
        label = clean_str(row.get("label"))
        category = clean_str(row.get("category"))
        text = clean_str(row.get("text"))
        if not text:
            continue
        canonical = "smishing"
        original = label or "phishing"
        if category:
            original = f"{original}:{category}"
        yield {
            "text": text,
            "canonical_label": canonical,
            "timestamp_original": None,
            "source": "kaggle_phishing",
            "source_id": str(i),
            "source_url": SOURCE_META["kaggle_phishing"]["source_url"],
            "original_label": original,
            "label_mapping_rule": "implicit_smishing",
            "license": SOURCE_META["kaggle_phishing"]["license"],
            "language_hint": None,
        }


def load_agarwal(path: Path, chunk_size: int) -> Iterable[Dict[str, Any]]:
    """Load Agarwal IMC dataset using text or translation field."""
    for i, row in enumerate(iter_csv_dicts(path, chunk_size=chunk_size), 1):
        text = clean_str(row.get("text")) or clean_str(row.get("translation"))
        if not text:
            continue
        language_hint = normalize_language(row.get("language"))
        yield {
            "text": text,
            "canonical_label": "smishing",
            "timestamp_original": clean_str(row.get("time")),
            "source": "agarwal_2025",
            "source_id": str(i),
            "source_url": SOURCE_META["agarwal_2025"]["source_url"],
            "original_label": "smishing",
            "label_mapping_rule": "implicit_smishing",
            "license": SOURCE_META["agarwal_2025"]["license"],
            "language_hint": language_hint,
        }


def load_uci(path: Path, chunk_size: int) -> Iterable[Dict[str, Any]]:
    """Load UCI SMS Spam Collection TSV (label, text)."""
    for i, row in enumerate(iter_csv_dicts(
        path,
        chunk_size=chunk_size,
        sep="\t",
        header=None,
        names=["label", "text"],
    ), 1):
        label = clean_str(row.get("label"))
        text = clean_str(row.get("text"))
        canonical = map_label_basic(label)
        if not canonical or not text:
            continue
        yield {
            "text": text,
            "canonical_label": canonical,
            "timestamp_original": None,
            "source": "uci_sms_spam",
            "source_id": str(i),
            "source_url": SOURCE_META["uci_sms_spam"]["source_url"],
            "original_label": label,
            "license": SOURCE_META["uci_sms_spam"]["license"],
            "language_hint": "en",
        }





def load_nus(path: Path) -> Iterable[Dict[str, Any]]:
    """Load the NUS SMS corpus from nested JSON."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        data = json.load(handle)
    messages = data.get("smsCorpus", {}).get("message", [])
    for i, msg in enumerate(messages, 1):
        text = msg.get("text", {})
        if isinstance(text, dict):
            text = text.get("$")
        text = clean_str(text)
        if not text:
            continue
        source_id = clean_str(msg.get("@id")) or str(i)
        yield {
            "text": text,
            "canonical_label": "ham",
            "timestamp_original": None,
            "source": "nus_sms",
            "source_id": source_id,
            "source_url": SOURCE_META["nus_sms"]["source_url"],
            "original_label": "ham",
            "label_mapping_rule": "implicit_ham",
            "license": SOURCE_META["nus_sms"]["license"],
            "language_hint": "en",
        }

def load_spanish(path: Path, chunk_size: int) -> Iterable[Dict[str, Any]]:
    """Load the Spanish spam/ham dataset with mensaje/tipo columns.

    Note: train.csv has a leading-space column name ' tipo' (with space).
    We fall back to the space-prefixed key to handle both variants.
    """
    for i, row in enumerate(iter_csv_dicts(path, chunk_size=chunk_size), 1):
        # train.csv has ' tipo' (leading space); test.csv has 'tipo' — handle both
        label = clean_str(row.get("tipo") or row.get(" tipo"))
        text = clean_str(row.get("mensaje"))
        canonical = map_label_basic(label)
        if not canonical or not text:
            continue
        yield {
            "text": text,
            "canonical_label": canonical,
            "timestamp_original": None,
            "source": "spanish_spam_ham",
            "source_id": str(i),
            "source_url": SOURCE_META["spanish_spam_ham"]["source_url"],
            "original_label": label,
            "license": SOURCE_META["spanish_spam_ham"]["license"],
            "language_hint": "es",
        }


def load_exais(path: Path) -> Iterable[Dict[str, Any]]:
    """Load ExAIS CSV files (Android SMS Backup & Restore format).

    The format is fixed-position:
      col 0: type (SMS/MMS)
      col 1: direction (send/receive)
      col 2-3: address numbers
      col 4: timestamp  (dd/mm/yyyy HH:MM)
      col 5: thread ID
      col 6: label (SPAM/HAM)
      col 7: message text  ← preferred extraction target
      col 8+: continuation cells for long messages split by CSV parser

    We prefer col 7 for text and fall back to the max-letters heuristic
    only when col 7 is empty (guards against edge-case CSV splits).
    """
    def score_cell(cell: str) -> int:
        return sum(ch.isalpha() for ch in cell)

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for i, row in enumerate(reader, 1):
            if not row:
                continue
            label = None
            for cell in row:
                if cell.strip().lower() in {"spam", "ham"}:
                    label = cell.strip().lower()
                    break
            if not label:
                continue
            # Column 7 is the message body in the EXAIS fixed format.
            # Concatenate col 7+ to recover messages split across cells.
            if len(row) > 7:
                text = " ".join(c.strip() for c in row[7:] if c.strip())
            else:
                # Fallback: pick the cell with the most alphabetic characters
                candidates = [c.strip() for c in row if c.strip()]
                text = max(candidates, key=score_cell) if candidates else None
            text = clean_str(text) if text else None
            if not text:
                continue
            timestamp_original = None
            for cell in row:
                if "/" in cell and ":" in cell:
                    timestamp_original = cell.strip()
                    break
            canonical = map_label_basic(label)
            if not canonical:
                continue
            yield {
                "text": text,
                "canonical_label": canonical,
                "timestamp_original": timestamp_original,
                "source": "exais_sms",
                "source_id": str(i),
                "source_url": SOURCE_META["exais_sms"]["source_url"],
                "original_label": label,
                "license": SOURCE_META["exais_sms"]["license"],
                "language_hint": "en",
            }


def load_malicious_benign(path: Path, chunk_size: int) -> Iterable[Dict[str, Any]]:
    """Load malicious/benign dataset variants with message/label fields."""
    for i, row in enumerate(iter_csv_dicts(path, chunk_size=chunk_size, usecols=["message", "label", "ai_generated"]), 1):
        text = clean_str(row.get("message"))
        label = clean_str(row.get("label"))
        ai_generated = clean_str(row.get("ai_generated"))
        canonical = map_binary_label(label, positive_label="spam")
        if not canonical or not text:
            continue
            
        rule_applied = f"label=={label}->{canonical}"
        
        yield {
            "text": text,
            "canonical_label": canonical,
            "timestamp_original": None,
            "source": "malicious_benign_sms_mms",
            "source_id": str(i),
            "source_url": SOURCE_META["malicious_benign_sms_mms"]["source_url"],
            "original_label": f"label={label};ai_generated={ai_generated}",
            "label_mapping_rule": rule_applied,
            "license": SOURCE_META["malicious_benign_sms_mms"]["license"],
            "language_hint": "en",
            "is_ai_generated": int(ai_generated) if ai_generated and ai_generated.strip() in {"1", "true"} else 0,
        }



def load_smishing_4c(path: Path, chunk_size: int) -> Iterable[Dict[str, Any]]:
    """Load Smishing-4C dataset with TYPE category and feature columns."""
    feature_cols = ["SLANG", "COMPANY", "Length_value", "Num_writing_errors", "Phone", "URL"]
    for i, row in enumerate(iter_csv_dicts(path, chunk_size=chunk_size), 1):
        text = clean_str(row.get("TEXT-ENG"))
        stype = clean_str(row.get("TYPE")) or "unknown"
        if not text:
            continue
        features = {col: clean_str(row.get(col)) or "" for col in feature_cols}
        features_str = ";".join(f"{k}={v}" for k, v in features.items())
        yield {
            "text": text,
            "canonical_label": "smishing",
            "timestamp_original": None,
            "source": "smishing_4c",
            "source_id": str(i),
            "source_url": SOURCE_META["smishing_4c"]["source_url"],
            "original_label": f"{stype} [{features_str}]",
            "label_mapping_rule": "implicit_smishing",
            "license": SOURCE_META["smishing_4c"]["license"],
            "language_hint": "en",
        }


def load_mimics_3500(path: Path, chunk_size: int) -> Iterable[Dict[str, Any]]:
    """Load MIMICS-3500 multi-class smishing dataset.

    Note: the file uses latin-1 encoding (not UTF-8); passing it explicitly
    avoids silent character corruption via the 'replace' error handler.
    """
    for i, row in enumerate(iter_csv_dicts(path, chunk_size=chunk_size, encoding="latin-1"), 1):
        text = clean_str(row.get("TEXT"))
        cls7 = clean_str(row.get("7_CLASSES")) or "unknown"
        cls13 = clean_str(row.get("13_CLASSES")) or "unknown"
        dataset = clean_str(row.get("DATASET")) or "unknown"
        if not text:
            continue
        yield {
            "text": text,
            "canonical_label": "smishing",
            "timestamp_original": None,
            "source": "mimics_3500",
            "source_id": str(i),
            "source_url": SOURCE_META["mimics_3500"]["source_url"],
            "original_label": f"7class={cls7};13class={cls13};dataset={dataset}",
            "license": SOURCE_META["mimics_3500"]["license"],
            "language_hint": "en",
        }


def build_metasms_dataset(
    raw_dir: Path,
    output_csv: Path,
    use_privacy_filter: bool,
    privacy_filter_model: str,
    dedupe: bool,
    resume: bool,
    checkpoint_every: int,
    chunk_size: int,
    exclude_ai_generated: bool = True,
) -> None:
    """Build the MetaSMS-HSS dataset from all raw sources.

    Args:
        exclude_ai_generated: When True (default), rows where ``is_ai_generated``
            is truthy are silently dropped before writing.  Pass False to keep
            them (e.g. for experiments that deliberately include synthetic data).
    """
    sources = []

    sources.append({
        "name": "mishra_soni_2022",
        "path": raw_dir / "mishra_soni_20_06_2022.csv",
        "loader": lambda p: load_mishra(p, "mishra_soni_2022", chunk_size),
    })
    sources.append({
        "name": "mishra_extended",
        "path": raw_dir / "mishra_extended_10191.csv",
        "loader": lambda p: load_mishra(p, "mishra_extended", chunk_size),
    })
    sources.append({
        "name": "hosseinpour_2025",
        "path": raw_dir / "hosseinpour_2025_combined.csv",
        "loader": lambda p: load_hosseinpour(p, chunk_size),
    })
    sources.append({
        "name": "kaggle_spam_ham",
        "path": raw_dir / "kaggle_unknown_combined_spam_ham.csv",
        "loader": lambda p: load_kaggle_spam_ham(p, chunk_size),
    })
    sources.append({
        "name": "kaggle_phishing",
        "path": raw_dir / "kaggle_unknown_phishing_categories.csv",
        "loader": lambda p: load_kaggle_phishing(p, chunk_size),
    })
    sources.append({
        "name": "agarwal_2025",
        "path": raw_dir / "agarwal_2025_fishing_smishing.csv",
        "loader": lambda p: load_agarwal(p, chunk_size),
    })
    sources.append({
        "name": "uci_sms_spam",
        "path": raw_dir / "almeida_2011_uci_sms_spam" / "SMSSpamCollection",
        "loader": lambda p: load_uci(p, chunk_size),
    })

    sources.append({
        "name": "nus_sms",
        "path": raw_dir / "nus_2015_sms_corpus_en.json",
        "loader": lambda p: load_nus(p),
    })

    spanish_dir = raw_dir / "spanish_spam_ham"
    for name in ["train.csv", "test.csv"]:
        sources.append({
            "name": f"spanish_spam_ham_{name}",
            "path": spanish_dir / name,
            # Default arg captures current value of chunk_size (not a loop var, but
            # explicit default arg is the canonical Python pattern for lambda closures).
            "loader": lambda p, cs=chunk_size: load_spanish(p, cs),
        })

    exais_dir = raw_dir / "onashoga_2015_exais_sms_spam"
    for exais_file in sorted(exais_dir.glob("USER *.csv")):
        sources.append({
            "name": f"exais_sms_{exais_file.stem}",
            "path": exais_file,
            # Capture exais_file by value via default arg to avoid the classic
            # Python lambda-in-loop late-binding bug (all lambdas would otherwise
            # share the same final value of exais_file after the loop ends).
            "loader": lambda p, _f=exais_file: load_exais(_f),
        })

    malicious_dir = raw_dir / "malicious_benign_sms_mms"
    for filename in [
        "dataset_v3_undersampled_stratified_full.csv",
    ]:
        sources.append({
            "name": f"malicious_benign_{filename}",
            "path": malicious_dir / filename,
            "loader": lambda p: load_malicious_benign(p, chunk_size),
        })

    sources.append({
        "name": "smishing_4c",
        "path": raw_dir / "smishing_4C.csv",
        "loader": lambda p: load_smishing_4c(p, chunk_size),
    })

    sources.append({
        "name": "mimics_3500",
        "path": raw_dir / "mimics_3500_v1.csv",
        "loader": lambda p: load_mimics_3500(p, chunk_size),
    })



    cleaner = DatasetCleaner()
    anonymizer = SMSAnonymizer(
        use_privacy_filter=use_privacy_filter,
        privacy_filter_model=privacy_filter_model,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    state_path = output_csv.with_suffix(".state.json")
    hashes_path = output_csv.with_suffix(".hashes.txt")
    state = load_resume_state(state_path) if resume else {}
    if "sources" not in state:
        state["sources"] = {}
    signature = state_signature(
        raw_dir,
        use_privacy_filter,
        privacy_filter_model,
        dedupe,
        chunk_size,
        exclude_ai_generated,
    )

    if resume and state:
        if state.get("options") != signature:
            logger.error("State options mismatch.")
            logger.error(f"Expected (from state.json): {state.get('options')}")
            logger.error(f"Got (from arguments): {signature}")
            raise ValueError("Resume options do not match the existing state file.")

    # Cache of hashes to prevent duplicate rows across sources.
    seen_hashes = set()
    if dedupe and hashes_path.exists():
        with hashes_path.open("r", encoding="utf-8") as handle:
            seen_hashes = {line.strip() for line in handle if line.strip()}

    output_exists = output_csv.exists()
    open_mode = "a" if resume and output_exists else "w"

    if open_mode == "w":
        # If we are starting fresh, we MUST clear the old hashes and state 
        # so we don't falsely skip rows that aren't in the new CSV.
        if hashes_path.exists():
            hashes_path.unlink()
        if state_path.exists():
            state_path.unlink()
        seen_hashes = set()

    with output_csv.open(open_mode, encoding="utf-8", newline="") as handle, \
        hashes_path.open("a", encoding="utf-8") as hash_handle:
        writer = csv.DictWriter(handle, fieldnames=MASTER_FIELDS)
        if open_mode == "w":
            writer.writeheader()

        for source in sources:
            path = source["path"]
            if not path.exists():
                logger.warning("Source missing, skipped: %s", path)
                continue
            logger.info("Processing %s", source["name"])
            source_name = source["name"]
            source_count = int(state["sources"].get(source_name, {}).get("written", 0)) if resume else 0
            source_offset = int(state["sources"].get(source_name, {}).get("seen", 0)) if resume else 0
            current_seen = 0
            batch_raws = []

            for raw in source["loader"](path):
                current_seen += 1
                if resume and current_seen <= source_offset:
                    continue
                
                source_seen = current_seen

                # Drop AI-generated messages when the flag is active.
                if exclude_ai_generated and raw.get("is_ai_generated"):
                    continue

                batch_raws.append(raw)

                if len(batch_raws) >= 128:
                    builts = build_master_rows(batch_raws, cleaner, anonymizer)
                    for built in builts:
                        if not built:
                            continue
                        row, dedupe_key = built

                        if dedupe and dedupe_key:
                            if dedupe_key in seen_hashes:
                                continue
                            seen_hashes.add(dedupe_key)
                            hash_handle.write(dedupe_key + "\n")
                        
                        writer.writerow(row)
                        source_count += 1

                    batch_raws.clear()

                    # Periodic checkpoints allow safe resume after interruptions.
                    if checkpoint_every and source_seen % checkpoint_every < 128:
                        state["version"] = 1
                        state["options"] = signature
                        state["sources"][source_name] = {"seen": source_seen, "written": source_count}
                        state["total_written"] = sum(s.get("written", 0) for s in state["sources"].values())
                        save_resume_state(state_path, state)
                        handle.flush()
                        hash_handle.flush()

            # Process remaining rows in the batch
            if batch_raws:
                builts = build_master_rows(batch_raws, cleaner, anonymizer)
                for built in builts:
                    if not built:
                        continue
                    row, dedupe_key = built

                    if dedupe and dedupe_key:
                        if dedupe_key in seen_hashes:
                            continue
                        seen_hashes.add(dedupe_key)
                        hash_handle.write(dedupe_key + "\n")
                    
                    writer.writerow(row)
                    source_count += 1
                batch_raws.clear()
            
            logger.info("Finished %s: %d records", source["name"], source_count)

            state["version"] = 1
            state["options"] = signature
            state["sources"][source_name] = {"seen": source_seen if 'source_seen' in locals() else source_offset, "written": source_count}
            state["total_written"] = sum(s.get("written", 0) for s in state["sources"].values())
            save_resume_state(state_path, state)
            handle.flush()
            hash_handle.flush()

    final_total = state.get("total_written", 0)
    logger.info("MetaSMS-HSS dataset built successfully. Total records: %d", final_total)
    logger.info("Saved to: %s", output_csv)

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description="Build MetaSMS-HSS dataset from all raw sources")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw datasets root")
    parser.add_argument("--output", type=str, default="", help="Path to save output CSV (auto-named if empty)")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Row chunk size for CSV sources")
    parser.add_argument("--privacy-filter", action="store_true", help="Use privacy-filter for anonymization")
    parser.add_argument("--privacy-filter-model", type=str, default="openai/privacy-filter", help="Privacy filter model name")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable cross-dataset deduplication")
    parser.add_argument("--resume", action="store_true", help="Resume from previous partial run")
    parser.add_argument("--checkpoint-every", type=int, default=10000, help="Checkpoint every N rows per source")
    parser.add_argument(
        "--include-ai-generated",
        action="store_true",
        default=False,
        help=(
            "Include AI-generated messages in the output dataset. "
            "By default they are excluded to avoid synthetic bias in training."
        ),
    )

    args = parser.parse_args()
    exclude_ai = not args.include_ai_generated

    output_csv = Path(args.output) if args.output else build_output_name(
        output_dir=Path("data/processed"),
        use_privacy_filter=args.privacy_filter,
        dedupe=not args.no_dedupe,
        exclude_ai_generated=exclude_ai,
    )

    build_metasms_dataset(
        raw_dir=Path(args.raw_dir),
        output_csv=output_csv,
        use_privacy_filter=args.privacy_filter,
        privacy_filter_model=args.privacy_filter_model,
        dedupe=not args.no_dedupe,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        chunk_size=args.chunk_size,
        exclude_ai_generated=exclude_ai,
    )
