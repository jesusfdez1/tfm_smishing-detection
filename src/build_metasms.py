import os
import uuid
import pandas as pd
import logging
import argparse
from datetime import datetime

from pipeline import PreprocessingPipeline
from utils.llm_enricher import LLMMetadataAnnotator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def build_metasms_dataset(input_csv: str, output_csv: str, use_llm: bool = True):
    """
    Builds the MetaSMS-HSS unified dataset from a raw input CSV.
    Input CSV must have at least: 'text', 'label', 'source'
    """
    logger.info(f"Loading raw dataset from {input_csv}")
    if not os.path.exists(input_csv):
        logger.error(f"File not found: {input_csv}")
        # Let's create a dummy dataset on the fly for testing if not found
        logger.info("Creating dummy dataset for demonstration...")
        df = pd.DataFrame({
            "text": [
                "C\u200Bo\u200Br\u200Br\u200Be\u200Bo\u200Bs: Su paquete esta retenido. Pague en http://bit.ly/fake123",
                "Your Amazon package is out for delivery. Track: https://amazon.com/track",
                "Alerta Santander: Cuenta bloqueada por seguridad. Acceda ya: http://sntndr-security-verify.xyz",
                "Feliz cumpleaños Maria! Nos vemos a las 8"
            ],
            "label": ["phishing", "ham", "smishing", "ham"],
            "source": ["Smishtank", "MIMICS-3500", "CustomDB", "CustomDB"]
        })
    else:
        df = pd.read_csv(input_csv)

    # Initialize Modules
    preprocessing = PreprocessingPipeline(use_whois=False)
    llm_annotator = LLMMetadataAnnotator(use_mock=not use_llm) # Use mock if LLM is disabled to save time/money
    
    final_records = []
    
    logger.info("Starting processing pipeline...")
    for idx, row in df.iterrows():
        raw_text = str(row.get("text", ""))
        original_label = str(row.get("label", "unknown")).lower()
        source = str(row.get("source", "unknown"))
        
        # 1. Map to Canonical Label (ham, spam, smishing)
        canonical_label = "ham"
        mapping_rule = f"{original_label}->ham"
        if original_label in ["phishing", "smishing", "malicious", "fraud"]:
            canonical_label = "smishing"
            mapping_rule = f"{original_label}->smishing"
        elif original_label in ["spam", "promotional", "marketing"]:
            canonical_label = "spam"
            mapping_rule = f"{original_label}->spam"

        # 2. Run Preprocessing Pipeline (Cleaner + Anonymizer + URL Features)
        preprocessed_data = preprocessing.process_row(raw_text)
        cleaned_text = preprocessed_data["processed_text"]

        # 3. Run LLM Enrichment (on the CLEANED text to save tokens and protect PII)
        llm_data = llm_annotator.annotate(cleaned_text)

        # Assemble the 4-Block MetaSMS-HSS Structure
        record = {
            # Block 1: Core
            "message_id": f"msg_{uuid.uuid4().hex[:8]}",
            "text": raw_text,
            "canonical_label": canonical_label,
            "timestamp_original": None,
            
            # Block 2: Traceability
            "source": source,
            "source_id": None,
            "source_url": None,
            "original_label": original_label,
            "label_mapping_rule": mapping_rule,
            "license": "CC BY 4.0" if source == "Smishtank" else "Research Use",
            
            # Block 3: Linguistic
            "language": llm_data["language"],
            "language_confidence": llm_data["language_confidence"],
            
            # Block 4: LLM Enrichment
            "llm_annotated": llm_data["llm_annotated"],
            "llm_model": llm_data["llm_model"],
            "prompt_version": llm_data["prompt_version"],
            "llm_annotation_date": llm_data["llm_annotation_date"],
            "theme": llm_data["theme"],
            "urgency_level": llm_data["urgency_level"],
            
            # Auxiliary Feature Engineering Block (URL Lexical Parity fix)
            "clean_anonymized_text": cleaned_text,
            "feat_url_entropy": preprocessed_data["max_url_entropy"],
            "feat_is_suspicious_tld": preprocessed_data["has_suspicious_tld"],
            "feat_num_subdomains": preprocessed_data["max_num_subdomains"]
        }
        
        final_records.append(record)

    # Save to CSV
    final_df = pd.DataFrame(final_records)
    final_df.to_csv(output_csv, index=False)
    
    logger.info(f"MetaSMS-HSS dataset built successfully! Total records: {len(final_df)}")
    logger.info(f"Saved to: {output_csv}")
    
    # Print sample
    print("\n--- SAMPLE OUTPUT ---")
    print(final_df.head(2).T)

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    parser = argparse.ArgumentParser(description="Build MetaSMS-HSS dataset")
    parser.add_argument("--input", type=str, default="raw_data.csv", help="Path to raw CSV dataset")
    parser.add_argument("--output", type=str, default="metasms_hss_master.csv", help="Path to save output CSV")
    parser.add_argument("--no-llm", action="store_true", help="Disable real LLM API and use mock heuristic enrichment")
    
    args = parser.parse_args()
    
    build_metasms_dataset(args.input, args.output, use_llm=not args.no_llm)
