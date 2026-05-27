with open('src/build_metasms.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add SOURCE_META
content = content.replace('''    "malicious_benign_sms_mms": {
        "license": "CC BY-NC 4.0",
        "source_url": None,
    },
}''', '''    "malicious_benign_sms_mms": {
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
}''')

# 2. MASTER_FIELDS
content = content.replace('''    "language_confidence",
    "llm_annotated",''', '''    "language_confidence",
    "is_ai_generated",
    "llm_annotated",''')

# 3. dedupe_text skeleton
content = content.replace('''def normalize_dedupe_text(text: str, mode: str) -> str:
    """Normalize text for dedupe with selectable strictness."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\\r", " ").replace("\\n", " ").replace("\\t", " ")
    normalized = " ".join(normalized.split())
    normalized = normalized.casefold()
    if mode == "soft":
        normalized = re.sub(r"\\W+", "", normalized, flags=re.UNICODE)
    return normalized''', '''def normalize_dedupe_text(text: str, mode: str) -> str:
    """Normalize text for dedupe with selectable strictness."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\\r", " ").replace("\\n", " ").replace("\\t", " ")
    normalized = " ".join(normalized.split())
    normalized = normalized.casefold()
    
    if mode == "skeleton":
        # Remove URLs (including those without http:// like bit.ly/123)
        normalized = re.sub(r'https?://\\S+|www\\.\\S+|\\b[a-z0-9\\-]+\\.[a-z]{2,}(?:/[^\\s]*)?\\b', '', normalized)
        # Remove tags, placeholders, explicit obfuscation sequences
        normalized = re.sub(r'<[^>]+>|\\[[^\\]]+\\]|[x*_\\-]{3,}', '', normalized)
        # Remove all digits (e.g. tracking numbers, phones not caught by tags)
        normalized = re.sub(r'\\d+', '', normalized)
        # Strip everything else except standard letters
        normalized = re.sub(r'\\W+', '', normalized, flags=re.UNICODE)
        
        # Truncate to the first 80 chars of the skeleton (~10-15 words).
        normalized = normalized[:80]
    elif mode == "soft":
        normalized = re.sub(r"\\W+", "", normalized, flags=re.UNICODE)
        
    return normalized''')

# 4. save_resume_state
content = content.replace('''def save_resume_state(state_path: Path, state: Dict[str, Any]) -> None:
    """Persist resume state using a temp file for atomic writes."""
    tmp_path = state_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=True, indent=2)
    tmp_path.replace(state_path)''', '''def save_resume_state(state_path: Path, state: Dict[str, Any]) -> None:
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
            time.sleep(0.5)''')

# 5. build_master_row
content = content.replace('''        "language": language_data.get("language", "unknown"),
        "language_confidence": language_data.get("language_confidence", 0.0),
        "llm_annotated": 1 if llm_annotations else 0,''', '''        "language": language_data.get("language", "unknown"),
        "language_confidence": language_data.get("language_confidence", 0.0),
        "is_ai_generated": int(raw.get("is_ai_generated", 0)),
        "llm_annotated": 1 if llm_annotations else 0,''')

# 6. load_kaggle_phishing bug fix
content = content.replace('''canonical = "smishing" if label else "smishing"''', '''canonical = "smishing"''')

# 7. malicious_benign loaders
content = content.replace('''            "label_mapping_rule": "label==1->spam; label==0->ham",
            "license": SOURCE_META["malicious_benign_sms_mms"]["license"],
            "language_hint": "en",
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
        original = f"label={label};source={clean_str(row.get('source'))};model={clean_str(row.get('model'))}"
        yield {
            "text": text,
            "canonical_label": canonical,
            "timestamp_original": None,
            "source": "malicious_benign_sms_mms",
            "source_id": None,
            "source_url": SOURCE_META["malicious_benign_sms_mms"]["source_url"],
            "original_label": original,
            "label_mapping_rule": "label==1->spam; label==0->ham",
            "license": SOURCE_META["malicious_benign_sms_mms"]["license"],
            "language_hint": "en",
        }''', '''            "label_mapping_rule": "label==1->spam; label==0->ham",
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
        original = f"label={label};source={clean_str(row.get('source'))};model={clean_str(row.get('model'))}"
        yield {
            "text": text,
            "canonical_label": canonical,
            "timestamp_original": None,
            "source": "malicious_benign_synthetic",
            "source_id": None,
            "source_url": SOURCE_META["malicious_benign_sms_mms"]["source_url"],
            "original_label": original,
            "label_mapping_rule": "label==1->spam; label==0->ham",
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
        }''')

# 8. load_malicious_benign missing 'ai_generated' extraction
content = content.replace('''def load_malicious_benign(path: Path, chunk_size: int) -> Iterable[Dict[str, Any]]:
    """Load malicious/benign dataset variants with message/label fields."""
    for row in iter_csv_dicts(path, chunk_size=chunk_size, usecols=["message", "label"]):
        text = clean_str(row.get("message"))
        label = clean_str(row.get("label"))
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
            "original_label": label,''', '''def load_malicious_benign(path: Path, chunk_size: int) -> Iterable[Dict[str, Any]]:
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
            "original_label": f"label={label};ai_generated={ai_generated}",''')

# 9. Register new sources
content = content.replace('''    sources.append({
        "name": "malicious_benign_synthetic",
        "path": malicious_dir / "synthetic_data" / "ai_generated_all.csv",
        "loader": lambda p: load_malicious_benign_synthetic(p, chunk_size),
    })

    if sources_filter:''', '''    sources.append({
        "name": "malicious_benign_synthetic",
        "path": malicious_dir / "synthetic_data" / "ai_generated_all.csv",
        "loader": lambda p: load_malicious_benign_synthetic(p, chunk_size),
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

    if sources_filter:''')

# 10. CLI args
content = content.replace('''    parser.add_argument("--dedupe-mode", type=str, default="soft", choices=["strict", "soft"], help="Deduplication strictness: strict (whitespace only) or soft (ignore punctuation)")''', '''    parser.add_argument("--dedupe-mode", type=str, default="skeleton", choices=["strict", "soft", "skeleton"], help="Deduplication strictness: strict (whitespace only), soft (ignore punctuation), or skeleton (removes numbers, urls, tags)")''')

with open('src/build_metasms.py', 'w', encoding='utf-8') as f:
    f.write(content)
