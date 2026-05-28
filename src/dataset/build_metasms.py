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
from utils.llm_enricher import LLMMetadataAnnotator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Output schema for the master dataset.
MASTER_FIELDS = [
    "message_id",
    "text",
    "text_anonymized",
    "anonymization_status",
    "canonical_label",
    "timestamp_original",
    "source",
    "source_id",
    "source_url",
    "original_label",
    "label_mapping_rule",
    "license",
    "language",
    "language_confidence",
    "is_ai_generated",
    "llm_annotated",
    "llm_model",
    "prompt_version",
    "llm_annotation_date",
    "theme",
    "urgency_level",
]

LANGUAGE_MAP = {
    "portuguese": "pt",
    "dutch": "nl",
    "spanish": "es",
    "english": "en",
    "french": "fr",
    "german": "de",
    "italian": "it",
}

SOURCE_META = {
    "smishtank": {
        "license": "Academic Use Only",
        "source_url": "https://smishtank.com/",
    },
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
    "enron_spam": {
        "license": "Research Use",
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
    """Clean string by removing surrounding whitespace and internal newlines."""
    if not isinstance(value, str):
        return None
    # Replace carriage returns and newlines with a single space
    cleaned = value.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    # Collapse multiple spaces into one
    cleaned = " ".join(cleaned.split())
    return cleaned if cleaned else None


def normalize_language(value: Optional[str]) -> Optional[str]:
    """Normalize language names to ISO-639-1 codes when possible."""
    if not value:
        return None
    raw = str(value).strip().lower()
    if len(raw) == 2:
        return raw
    return LANGUAGE_MAP.get(raw)


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
    """Yield dict rows from a CSV using chunked reads for large files."""
    for chunk in pd.read_csv(
        path,
        chunksize=chunk_size,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        encoding_errors="replace",
        on_bad_lines="skip",
        **kwargs,
    ):
        for record in chunk.to_dict(orient="records"):
            yield record


def clean_for_llm(text: str, cleaner: DatasetCleaner, anonymizer: SMSAnonymizer) -> str:
    """Clean and anonymize text to reduce PII exposure in LLM calls."""
    cleaned = cleaner.clean(text).get("cleaned", text)
    return anonymizer.process_message(cleaned).get("anonymized", cleaned)


def detect_pre_anonymized(text: str) -> bool:
    """Detect placeholder tokens that indicate prior anonymization."""
    if not text:
        return False
    return bool(re.search(r"<[A-Z0-9_]+>", text))


def build_dedupe_key(text: str) -> str:
    """Hash a normalized text string for cross-source deduplication."""
    if not text:
        return ""
    # Normalize and hash the anonymized text for cross-source deduplication.
    normalized = unicodedata.normalize("NFKC", text)
    normalized = " ".join(normalized.lower().split())
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def sanitize_token(value: str) -> str:
    """Make a filename-safe token from arbitrary user input."""
    safe = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip())
    safe = safe.strip("-").lower()
    return safe or "na"


def build_output_name(
    output_dir: Path,
    use_llm: bool,
    use_label_llm: bool,
    llm_models: List[str],
    use_privacy_filter: bool,
    privacy_filter_model: str,
    dedupe: bool,
) -> Path:
    """Build an output filename that encodes the main pipeline options."""
    llm_flag = "on" if use_llm else "off"
    label_flag = "on" if use_label_llm else "off"
    privacy_flag = "on" if use_privacy_filter else "off"
    dedupe_flag = "on" if dedupe else "off"
    model_token = "none" if not use_llm else "+".join([sanitize_token(m) for m in llm_models])
    privacy_token = sanitize_token(privacy_filter_model) if use_privacy_filter else "none"
    name = (
        "metasms_hss_master"
        f"__llm={llm_flag}"
        f"__label={label_flag}"
        f"__models={model_token}"
        f"__privacy={privacy_flag}"
        f"__pmodel={privacy_token}"
        f"__dedupe={dedupe_flag}.csv"
    )
    return output_dir / name


def state_signature(
    raw_dir: Path,
    use_llm: bool,
    use_label_llm: bool,
    llm_models: List[str],
    use_privacy_filter: bool,
    privacy_filter_model: str,
    dedupe: bool,
    chunk_size: int,
) -> Dict[str, Any]:
    """Return a stable signature to validate resume compatibility."""
    return {
        "raw_dir": str(raw_dir),
        "use_llm": use_llm,
        "use_label_llm": use_label_llm,
        "llm_models": llm_models,
        "use_privacy_filter": use_privacy_filter,
        "privacy_filter_model": privacy_filter_model,
        "dedupe": dedupe,
        "chunk_size": chunk_size,
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
            from langdetect import detect_langs

            candidates = detect_langs(text)
            if candidates:
                detected = candidates[0].lang[:2].lower()
                confidence = float(candidates[0].prob)
        except Exception:
            pass

    if not detected:
        try:
            import langid

            detected, score = langid.classify(text)
            detected = detected[:2].lower()
            confidence = float(score)
        except Exception:
            pass

    if not detected:
        if hint:
            return {"language": hint, "language_confidence": 1.0}
        return {"language": "unknown", "language_confidence": 0.0}

    return {"language": detected, "language_confidence": confidence}


def llm_metadata(text: str, annotator: LLMMetadataAnnotator, use_llm: bool) -> Dict[str, Any]:
    """Call an LLM annotator for theme and urgency metadata."""
    if not use_llm:
        return {
            "llm_annotated": 0,
            "llm_model": None,
            "prompt_version": None,
            "llm_annotation_date": None,
            "theme": 0,
            "urgency_level": 0,
        }

    data = annotator.annotate(text)
    return {
        "llm_annotated": data.get("llm_annotated", 0),
        "llm_model": data.get("llm_model"),
        "prompt_version": data.get("prompt_version"),
        "llm_annotation_date": data.get("llm_annotation_date"),
        "theme": data.get("theme", 0),
        "urgency_level": data.get("urgency_level", 0),
    }


def collect_llm_annotations(
    text: str,
    annotators: Dict[str, LLMMetadataAnnotator],
    use_llm: bool,
) -> List[Dict[str, Any]]:
    """Run all configured LLM annotators and return raw results."""
    if not use_llm or not annotators:
        return []

    results: List[Dict[str, Any]] = []
    for model_name, annotator in annotators.items():
        data = llm_metadata(text, annotator, use_llm=True)
        data["llm_model"] = model_name
        results.append(data)
    return results


def llm_classify_canonical_label(text: str) -> Optional[str]:
    """Classify a message into ham/spam/smishing using an LLM."""
    # TODO: Implement the real LLM call here and return "ham", "spam", or "smishing".
    return None


def resolve_canonical_label(
    raw_label: Optional[str],
    use_label_llm: bool,
    label_text: str,
) -> Optional[str]:
    """Resolve a canonical label, optionally using an LLM for unlabeled rows."""
    if raw_label:
        return raw_label
    if not use_label_llm:
        return None
    predicted = llm_classify_canonical_label(label_text)
    if predicted in {"ham", "spam", "smishing"}:
        return predicted
    return None


def parse_llm_models(value: Optional[str]) -> List[str]:
    """Parse a comma-separated LLM list, falling back to a default model."""
    if not value:
        return [LLMMetadataAnnotator.MODEL_NAME]
    models = [model.strip() for model in value.split(",") if model.strip()]
    return models if models else [LLMMetadataAnnotator.MODEL_NAME]


def build_master_row(
    raw: Dict[str, Any],
    annotators: Dict[str, LLMMetadataAnnotator],
    cleaner: DatasetCleaner,
    anonymizer: SMSAnonymizer,
    use_llm: bool,
    use_label_llm: bool,
) -> Optional[Tuple[Dict[str, Any], str]]:
    """Build a normalized master row and return a dedupe key."""
    text = clean_str(raw.get("text"))
    if not text:
        return None

    # Use the anonymized text for language detection, dedupe, and LLM tasks.
    llm_text = clean_for_llm(text, cleaner, anonymizer)
    if detect_pre_anonymized(text):
        anonymization_status = "pre_anonymized"
    elif llm_text != text:
        anonymization_status = "anonymized"
    else:
        anonymization_status = "raw"
    canonical_label = resolve_canonical_label(raw.get("canonical_label"), use_label_llm, llm_text)
    if not canonical_label:
        logger.warning("Skipped unlabeled row from source=%s", raw.get("source"))
        return None

    language_data = detect_language(llm_text, raw.get("language_hint"))
    llm_annotations = collect_llm_annotations(llm_text, annotators, use_llm)
    llm_primary = llm_annotations[0] if llm_annotations else {}
    llm_models = ",".join(
        [entry.get("llm_model", "") for entry in llm_annotations if entry.get("llm_model")]
    )
    llm_annotations_json = json.dumps(llm_annotations, ensure_ascii=True)
    dedupe_key = build_dedupe_key(llm_text)

    original_label = raw.get("original_label") or "unlabeled"
    
    # If the row had no label and the LLM inferred it
    if not raw.get("canonical_label") and canonical_label:
        label_mapping_rule = f"{original_label}->{canonical_label} (LLM Inference)"
    else:
        # Standardize the mapping rule format directly from the values
        label_mapping_rule = f"{original_label}->{canonical_label}"

    return {
        "message_id": f"msg_{uuid.uuid4().hex[:12]}",
        "text": text,
        "text_anonymized": llm_text,
        "anonymization_status": anonymization_status,
        "canonical_label": canonical_label,
        "timestamp_original": raw.get("timestamp_original") or "",
        "source": raw.get("source") or "unknown",
        "source_id": raw.get("source_id") or "",
        "source_url": raw.get("source_url") or "",
        "original_label": original_label,
        "label_mapping_rule": label_mapping_rule,
        "license": raw.get("license") or "",
        "language": language_data.get("language", "unknown"),
        "language_confidence": language_data.get("language_confidence", 0.0),
        "is_ai_generated": int(raw.get("is_ai_generated", 0)),
        "llm_annotated": 1 if llm_annotations else 0,
        "llm_model": raw.get("llm_model") or llm_primary.get("llm_model") or "",
        "prompt_version": llm_primary.get("prompt_version") or "",
        "llm_annotation_date": llm_primary.get("llm_annotation_date") or "",
        "theme": llm_primary.get("theme", 0),
        "urgency_level": llm_primary.get("urgency_level", 0),
    }, dedupe_key


def load_mishra(path: Path, source_name: str, chunk_size: int) -> Iterable[Dict[str, Any]]:
    """Load Mishra/Soni sources with LABEL/TEXT columns."""
    for row in iter_csv_dicts(path, chunk_size=chunk_size):
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
            "source_id": None,
            "source_url": SOURCE_META[source_name]["source_url"],
            "original_label": label,
            "label_mapping_rule": f"{label}->{canonical}",
            "license": SOURCE_META[source_name]["license"],
            "language_hint": None,
        }


def load_hosseinpour(path: Path, chunk_size: int) -> Iterable[Dict[str, Any]]:
    """Load Hosseinpour dataset with spam and smishing binary labels."""
    for row in iter_csv_dicts(path, chunk_size=chunk_size):
        text = clean_str(row.get("message"))
        spam_label = clean_str(row.get("spam label") or row.get("spam_label"))
        smish_label = clean_str(row.get("smishing label") or row.get("smishing_label"))
        if not text:
            continue
        canonical = None
        if smish_label and smish_label.strip() == "1":
            canonical = "smishing"
        elif spam_label and spam_label.strip() == "1":
            canonical = "spam"
        else:
            canonical = "ham"
        yield {
            "text": text,
            "canonical_label": canonical,
            "timestamp_original": None,
            "source": "hosseinpour_2025",
            "source_id": None,
            "source_url": SOURCE_META["hosseinpour_2025"]["source_url"],
            "original_label": f"spam_label={spam_label};smishing_label={smish_label}",
            "label_mapping_rule": "smishing_label==1->smishing; spam_label==1->spam; else->ham",
            "license": SOURCE_META["hosseinpour_2025"]["license"],
            "language_hint": None,
        }


def load_kaggle_spam_ham(path: Path, chunk_size: int) -> Iterable[Dict[str, Any]]:
    """Load Kaggle spam/ham dataset with target/text columns."""
    for row in iter_csv_dicts(path, chunk_size=chunk_size):
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
            "source_id": None,
            "source_url": SOURCE_META["kaggle_spam_ham"]["source_url"],
            "original_label": label,
            "label_mapping_rule": f"{label}->{canonical}",
            "license": SOURCE_META["kaggle_spam_ham"]["license"],
            "language_hint": None,
        }


def load_kaggle_phishing(path: Path, chunk_size: int) -> Iterable[Dict[str, Any]]:
    """Load Kaggle phishing dataset and map to smishing."""
    for row in iter_csv_dicts(path, chunk_size=chunk_size):
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
            "source_id": None,
            "source_url": SOURCE_META["kaggle_phishing"]["source_url"],
            "original_label": original,
            "label_mapping_rule": "phishing->smishing",
            "license": SOURCE_META["kaggle_phishing"]["license"],
            "language_hint": None,
        }


def load_agarwal(path: Path, chunk_size: int) -> Iterable[Dict[str, Any]]:
    """Load Agarwal IMC dataset using text or translation field."""
    for row in iter_csv_dicts(path, chunk_size=chunk_size):
        text = clean_str(row.get("text")) or clean_str(row.get("translation"))
        if not text:
            continue
        language_hint = normalize_language(row.get("language"))
        yield {
            "text": text,
            "canonical_label": "smishing",
            "timestamp_original": clean_str(row.get("time")),
            "source": "agarwal_2025",
            "source_id": None,
            "source_url": SOURCE_META["agarwal_2025"]["source_url"],
            "original_label": "smishing",
            "label_mapping_rule": "implicit_smishing",
            "license": SOURCE_META["agarwal_2025"]["license"],
            "language_hint": language_hint,
        }


def load_uci(path: Path, chunk_size: int) -> Iterable[Dict[str, Any]]:
    """Load UCI SMS Spam Collection TSV (label, text)."""
    for row in iter_csv_dicts(
        path,
        chunk_size=chunk_size,
        sep="\t",
        header=None,
        names=["label", "text"],
    ):
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
            "source_id": None,
            "source_url": SOURCE_META["uci_sms_spam"]["source_url"],
            "original_label": label,
            "label_mapping_rule": f"{label}->{canonical}",
            "license": SOURCE_META["uci_sms_spam"]["license"],
            "language_hint": "en",
        }


def load_enron(path: Path, chunk_size: int) -> Iterable[Dict[str, Any]]:
    """Load Enron spam/ham emails and join subject/body."""
    for row in iter_csv_dicts(path, chunk_size=chunk_size):
        label = clean_str(row.get("Spam/Ham"))
        subject = clean_str(row.get("Subject"))
        message = clean_str(row.get("Message"))
        text_parts = [part for part in [subject, message] if part]
        text = "\n".join(text_parts)
        canonical = map_label_basic(label)
        if not canonical or not text:
            continue
        yield {
            "text": text,
            "canonical_label": canonical,
            "timestamp_original": clean_str(row.get("Date")),
            "source": "enron_spam",
            "source_id": clean_str(row.get("Message ID")),
            "source_url": SOURCE_META["enron_spam"]["source_url"],
            "original_label": label,
            "label_mapping_rule": f"{label}->{canonical}",
            "license": SOURCE_META["enron_spam"]["license"],
            "language_hint": "en",
        }


def load_nus(path: Path) -> Iterable[Dict[str, Any]]:
    """Load the NUS SMS corpus from nested JSON."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        data = json.load(handle)
    messages = data.get("smsCorpus", {}).get("message", [])
    for msg in messages:
        text = msg.get("text", {})
        if isinstance(text, dict):
            text = text.get("$")
        text = clean_str(text)
        if not text:
            continue
        source_id = clean_str(msg.get("@id"))
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


def load_smish(path: Path) -> Iterable[Dict[str, Any]]:
    """Load Smishtank JSONL records and map to smishing or leave unlabeled based on community evaluation."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            smish = payload.get("raw", {}).get("smish", {})
            text = smish.get("messageContent") or smish.get("rawText")
            text = clean_str(text)
            if not text:
                continue
            ts = smish.get("timeSubmitted") or smish.get("timeReceived")
            timestamp_original = iso_from_epoch_ms(ts)
            source_id = clean_str(smish.get("messageID") or payload.get("message_id"))
            source_url = clean_str(smish.get("url"))
            
            # Comprobar si la comunidad lo ha verificado
            upvotes = int(smish.get("upvotes") or 0)
            
            if upvotes > 0:
                canonical_label = "smishing"
                original_label = "smishing (community verified)"
            else:
                canonical_label = None
                original_label = "unlabeled (raw submission)"

            yield {
                "text": text,
                "canonical_label": canonical_label,
                "timestamp_original": timestamp_original,
                "source": "smishtank",
                "source_id": source_id,
                "source_url": source_url or SOURCE_META["smishtank"]["source_url"],
                "original_label": original_label,
                "label_mapping_rule": "", # Será sobreescrito dinámicamente en build_master_row
                "license": SOURCE_META["smishtank"]["license"],
                "language_hint": None,
            }


def load_spanish(path: Path, chunk_size: int) -> Iterable[Dict[str, Any]]:
    """Load the Spanish spam/ham dataset with mensaje/tipo columns."""
    for row in iter_csv_dicts(path, chunk_size=chunk_size):
        label = clean_str(row.get("tipo"))
        text = clean_str(row.get("mensaje"))
        canonical = map_label_basic(label)
        if not canonical or not text:
            continue
        yield {
            "text": text,
            "canonical_label": canonical,
            "timestamp_original": None,
            "source": "spanish_spam_ham",
            "source_id": None,
            "source_url": SOURCE_META["spanish_spam_ham"]["source_url"],
            "original_label": label,
            "label_mapping_rule": f"{label}->{canonical}",
            "license": SOURCE_META["spanish_spam_ham"]["license"],
            "language_hint": "es",
        }


def load_exais(path: Path) -> Iterable[Dict[str, Any]]:
    """Load ExAIS CSV files with variable column layouts."""
    def score_cell(cell: str) -> int:
        letters = sum(ch.isalpha() for ch in cell)
        return letters

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            label = None
            for cell in row:
                if cell.strip().lower() in {"spam", "ham"}:
                    label = cell.strip().lower()
                    break
            if not label:
                continue
            text_candidates = [cell.strip() for cell in row if cell.strip()]
            text = None
            if text_candidates:
                text = max(text_candidates, key=score_cell)
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
                "source_id": None,
                "source_url": SOURCE_META["exais_sms"]["source_url"],
                "original_label": label,
                "label_mapping_rule": f"{label}->{canonical}",
                "license": SOURCE_META["exais_sms"]["license"],
                "language_hint": "en",
            }


def load_malicious_benign(path: Path, chunk_size: int) -> Iterable[Dict[str, Any]]:
    """Load malicious/benign dataset variants with message/label fields."""
    for row in iter_csv_dicts(path, chunk_size=chunk_size, usecols=["message", "label", "ai_generated"]):
        text = clean_str(row.get("message"))
        label = clean_str(row.get("label"))
        ai_generated = clean_str(row.get("ai_generated"))
        canonical = map_binary_label(label, positive_label="spam")
        if not canonical or not text:
            continue
        yield {
            "text": text,
            "canonical_label": canonical,
            "timestamp_original": None,
            "source": "malicious_benign_sms_mms",
            "source_id": None,
            "source_url": SOURCE_META["malicious_benign_sms_mms"]["source_url"],
            "original_label": f"label={label};ai_generated={ai_generated}",
            "label_mapping_rule": "label==1->spam; label==0->ham",
            "license": SOURCE_META["malicious_benign_sms_mms"]["license"],
            "language_hint": "en",
            "is_ai_generated": int(ai_generated) if ai_generated and ai_generated.strip() in {"1", "true"} else 0,
        }


def load_malicious_benign_synthetic(path: Path, chunk_size: int) -> Iterable[Dict[str, Any]]:
    """Load synthetic smishing records from the generator outputs."""
    for row in iter_csv_dicts(path, chunk_size=chunk_size):
        text = clean_str(row.get("message"))
        label = clean_str(row.get("label"))
        if not text:
            continue
        canonical = map_binary_label(label, positive_label="spam")
        if not canonical:
            continue
        original = f"label={label};source={clean_str(row.get('source'))}"
        ai_model = clean_str(row.get("model"))
        yield {
            "text": text,
            "canonical_label": canonical,
            "timestamp_original": None,
            "source": "malicious_benign_synthetic",
            "source_id": None,
            "source_url": SOURCE_META["malicious_benign_sms_mms"]["source_url"],
            "original_label": original,
            "label_mapping_rule": "label==1->spam; label==0->ham",
            "llm_model": ai_model,
            "license": SOURCE_META["malicious_benign_sms_mms"]["license"],
            "language_hint": "en",
            "is_ai_generated": 1,
        }


def load_smishing_4c(path: Path, chunk_size: int) -> Iterable[Dict[str, Any]]:
    """Load Smishing-4C dataset with TYPE category and feature columns."""
    feature_cols = ["SLANG", "COMPANY", "Length_value", "Num_writing_errors", "Phone", "URL"]
    for row in iter_csv_dicts(path, chunk_size=chunk_size):
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
            "source_id": None,
            "source_url": SOURCE_META["smishing_4c"]["source_url"],
            "original_label": f"{stype} [{features_str}]",
            "label_mapping_rule": "implicit_smishing",
            "license": SOURCE_META["smishing_4c"]["license"],
            "language_hint": "en",
        }


def load_mimics_3500(path: Path, chunk_size: int) -> Iterable[Dict[str, Any]]:
    """Load MIMICS-3500 multi-class smishing dataset."""
    for row in iter_csv_dicts(path, chunk_size=chunk_size):
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
            "source_id": None,
            "source_url": SOURCE_META["mimics_3500"]["source_url"],
            "original_label": f"7class={cls7};13class={cls13};dataset={dataset}",
            "label_mapping_rule": "implicit_smishing",
            "license": SOURCE_META["mimics_3500"]["license"],
            "language_hint": "en",
        }


def build_metasms_dataset(
    raw_dir: Path,
    output_csv: Path,
    use_llm: bool,
    use_label_llm: bool,
    llm_models: List[str],
    use_privacy_filter: bool,
    privacy_filter_model: str,
    dedupe: bool,
    resume: bool,
    checkpoint_every: int,
    chunk_size: int,
) -> None:
    """Orchestrate the full build with optional resume and dedupe."""
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
    sources.append({
        "name": "smishtank",
        "path": raw_dir / "smishtank_20_05_2026.jsonl",
        "loader": lambda p: load_smish(p),
    })

    spanish_dir = raw_dir / "spanish_spam_ham"
    for name in ["train.csv", "test.csv"]:
        sources.append({
            "name": f"spanish_spam_ham_{name}",
            "path": spanish_dir / name,
            "loader": lambda p: load_spanish(p, chunk_size),
        })

    exais_dir = raw_dir / "onashoga_2015_exais_sms_spam"
    for exais_file in sorted(exais_dir.glob("USER *.csv")):
        sources.append({
            "name": f"exais_sms_{exais_file.stem}",
            "path": exais_file,
            "loader": lambda p: load_exais(p),
        })

    malicious_dir = raw_dir / "malicious_benign_sms_mms"
    for filename in [
        "dataset_v3_for_deberta.csv",
    ]:
        sources.append({
            "name": f"malicious_benign_{filename}",
            "path": malicious_dir / filename,
            "loader": lambda p: load_malicious_benign(p, chunk_size),
        })
    sources.append({
        "name": "malicious_benign_synthetic",
        "path": malicious_dir / "synthetic_data" / "ai_generated_all.csv",
        "loader": lambda p: load_malicious_benign_synthetic(p, chunk_size),
    })

    cleaner = DatasetCleaner()
    anonymizer = SMSAnonymizer(
        use_whois=False,
        use_privacy_filter=use_privacy_filter,
        privacy_filter_model=privacy_filter_model,
    )
    annotators = {
        model: LLMMetadataAnnotator(use_mock=not use_llm, model_name=model)
        for model in llm_models
    }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    total_written = 0

    state_path = output_csv.with_suffix(".state.json")
    hashes_path = output_csv.with_suffix(".hashes.txt")
    state = load_resume_state(state_path) if resume else {}
    signature = state_signature(
        raw_dir,
        use_llm,
        use_label_llm,
        llm_models,
        use_privacy_filter,
        privacy_filter_model,
        dedupe,
        chunk_size,
    )

    if resume and state:
        if state.get("options") != signature:
            raise ValueError("Resume options do not match the existing state file.")

    # Cache of hashes to prevent duplicate rows across sources.
    seen_hashes = set()
    if dedupe and hashes_path.exists():
        with hashes_path.open("r", encoding="utf-8") as handle:
            seen_hashes = {line.strip() for line in handle if line.strip()}

    output_exists = output_csv.exists()
    open_mode = "a" if resume and output_exists else "w"

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
            source_count = 0
            source_seen = 0
            source_name = source["name"]
            source_offset = 0
            if resume:
                source_offset = int(state.get("sources", {}).get(source_name, {}).get("seen", 0))

            for raw in source["loader"](path):
                source_seen += 1
                if resume and source_seen <= source_offset:
                    continue

                built = build_master_row(raw, annotators, cleaner, anonymizer, use_llm, use_label_llm)
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

                # Periodic checkpoints allow safe resume after interruptions.
                if checkpoint_every and source_seen % checkpoint_every == 0:
                    state = {
                        "version": 1,
                        "options": signature,
                        "sources": {
                            **state.get("sources", {}),
                            source_name: {"seen": source_seen, "written": source_count},
                        },
                        "total_written": total_written + source_count,
                    }
                    save_resume_state(state_path, state)
            logger.info("Finished %s: %d records", source["name"], source_count)
            total_written += source_count

            state = {
                "version": 1,
                "options": signature,
                "sources": {
                    **state.get("sources", {}),
                    source_name: {"seen": source_seen, "written": source_count},
                },
                "total_written": total_written,
            }
            save_resume_state(state_path, state)

    logger.info("MetaSMS-HSS dataset built successfully. Total records: %d", total_written)
    logger.info("Saved to: %s", output_csv)

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    parser = argparse.ArgumentParser(description="Build MetaSMS-HSS dataset from all raw sources")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw datasets root")
    parser.add_argument("--output", type=str, default="", help="Path to save output CSV (auto-named if empty)")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Row chunk size for CSV sources")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM enrichment and skip API calls")
    parser.add_argument("--label-llm", action="store_true", help="Use LLM to label unlabeled rows")
    parser.add_argument("--llm-models", type=str, default="", help="Comma-separated list of LLM models")
    parser.add_argument("--privacy-filter", action="store_true", help="Use privacy-filter for anonymization")
    parser.add_argument("--privacy-filter-model", type=str, default="openai/privacy-filter", help="Privacy filter model name")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable cross-dataset deduplication")
    parser.add_argument("--resume", action="store_true", help="Resume from previous partial run")
    parser.add_argument("--checkpoint-every", type=int, default=10000, help="Checkpoint every N rows per source")
    
    args = parser.parse_args()
    
    llm_models = parse_llm_models(args.llm_models)
    output_csv = Path(args.output) if args.output else build_output_name(
        output_dir=Path("."),
        use_llm=not args.no_llm,
        use_label_llm=args.label_llm,
        llm_models=llm_models,
        use_privacy_filter=args.privacy_filter,
        privacy_filter_model=args.privacy_filter_model,
        dedupe=not args.no_dedupe,
    )

    build_metasms_dataset(
        raw_dir=Path(args.raw_dir),
        output_csv=output_csv,
        use_llm=not args.no_llm,
        use_label_llm=args.label_llm,
        llm_models=llm_models,
        use_privacy_filter=args.privacy_filter,
        privacy_filter_model=args.privacy_filter_model,
        dedupe=not args.no_dedupe,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        chunk_size=args.chunk_size,
    )
