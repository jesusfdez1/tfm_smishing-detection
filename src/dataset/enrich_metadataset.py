"""Enrich the MetaSMS-HSS dataset using a Large Language Model.

This script reads a pre-consolidated CSV dataset, identifies rows needing
labeling or metadata enrichment, processes them in batches via the LLM API,
and writes the enriched results to a new CSV.
"""

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Dict, Any, List

from utils.llm_enricher import LLMMetadataAnnotator
from utils.whois_enricher import WhoisEnricher

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def process_batch(
    batch_buffer: List[Dict[str, Any]],
    annotator: LLMMetadataAnnotator,
    whois_enricher: WhoisEnricher,
    use_label_llm: bool,
    writer: csv.DictWriter,
) -> int:
    """Process a batch of rows, querying the LLM in one go, and write to CSV."""
    if not batch_buffer:
        return 0
    
    # Pre-process texts and determine which need LLM
    llm_texts = []
    indices_to_annotate = []
    
    for i, row in enumerate(batch_buffer):
        llm_text = row.get("text_anonymized") or row.get("text")
        llm_texts.append(llm_text)
        
        needs_label = not row.get("canonical_label")
        should_call_llm = True  # In this enrichment phase, we enrich everything
        
        if should_call_llm:
            indices_to_annotate.append(i)

    # Call LLM in batch
    llm_results_map = {}
    if indices_to_annotate and annotator:
        texts_to_annotate = [llm_texts[i] for i in indices_to_annotate]
        batch_results = annotator.annotate_batch(texts_to_annotate)
        
        for idx, result in zip(indices_to_annotate, batch_results):
            llm_results_map[idx] = result
            
    # Update rows and write
    written_count = 0
    for i, row in enumerate(batch_buffer):
        llm_primary = llm_results_map.get(i, {})
        
        # Update metadata
        row["llm_annotated"] = 1 if llm_primary.get("llm_annotated") else 0
        row["llm_model"] = llm_primary.get("llm_model", "")
        row["prompt_version"] = llm_primary.get("prompt_version", "")
        row["llm_annotation_date"] = llm_primary.get("llm_annotation_date", "")
        row["theme"] = llm_primary.get("theme", 0)
        row["urgency_level"] = llm_primary.get("urgency_level", 0)
        
        # Add WHOIS enrichment
        if whois_enricher:
            import json
            try:
                entities_str = row.get("entities", "[]")
                if entities_str and entities_str.strip() != "[]":
                    entities = json.loads(entities_str)
                    modified = False
                    for entity in entities:
                        if entity.get("type") == "URL" and "value" in entity:
                            w_data = whois_enricher.get_whois_data(entity["value"])
                            if w_data:
                                entity["whois"] = w_data
                                modified = True
                    if modified:
                        row["entities"] = json.dumps(entities)
            except Exception as e:
                logger.debug(f"Failed to process WHOIS for row: {e}")
        
        

        # Update label if it was missing and we are allowed to infer it
        if not row.get("canonical_label") and use_label_llm:
            predicted = llm_primary.get("label")
            if predicted in {"ham", "spam", "smishing"}:
                row["canonical_label"] = predicted
                row["label_inferred_by_ai"] = 1
                
                original_label = row.get("original_label") or "unlabeled"
                row["label_mapping_rule"] = f"{original_label}->{predicted} (LLM Inference)"

        writer.writerow(row)
        written_count += 1
        
    return written_count


def enrich_dataset(
    input_csv: Path,
    output_csv: Path,
    model_name: str,
    batch_size: int,
    use_label_llm: bool,
) -> None:
    """Read un-annotated dataset and run batch LLM and WHOIS enrichment."""
    if not input_csv.exists():
        logger.error("Input file not found: %s", input_csv)
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    annotator = LLMMetadataAnnotator(use_mock=False, model_name=model_name)
    whois_enricher = WhoisEnricher()
    
    total_written = 0
    batch_buffer = []

    with input_csv.open("r", encoding="utf-8") as in_handle, \
         output_csv.open("w", encoding="utf-8", newline="") as out_handle:
        
        reader = csv.DictReader(in_handle)
        if not reader.fieldnames:
            logger.error("No fieldnames found in input CSV")
            return
            
        writer = csv.DictWriter(out_handle, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            batch_buffer.append(row)
            if len(batch_buffer) >= batch_size:
                total_written += process_batch(
                    batch_buffer, annotator, whois_enricher, use_label_llm, writer
                )
                batch_buffer.clear()
                logger.info("Enriched %d rows...", total_written)
                
                # Respect 15 RPM limit for Gemini 3.1 Flash Lite (Free Tier)
                time.sleep(4)
                
        # Process remaining
        if batch_buffer:
            total_written += process_batch(
                batch_buffer, annotator, whois_enricher, use_label_llm, writer
            )
            batch_buffer.clear()
            logger.info("Enriched %d rows...", total_written)

    logger.info("LLM Enrichment completed successfully. Total records: %d", total_written)
    logger.info("Saved to: %s", output_csv)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    parser = argparse.ArgumentParser(description="Enrich MetaSMS-HSS dataset using LLM")
    parser.add_argument("--input", type=str, required=True, help="Path to input unlabeled CSV")
    parser.add_argument("--output", type=str, required=True, help="Path to save enriched output CSV")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="LLM model name to use. E.g. 'gemini-2.5-flash', 'llama-3.1-8b-instant', 'llama-3.3-70b-versatile'. If it starts with 'llama', it will use the Groq provider.")
    parser.add_argument("--batch-size", type=int, default=30, help="Number of rows to process in one LLM call")
    parser.add_argument("--label-llm", action="store_true", help="Use LLM to infer canonical_label for unlabeled rows")
    
    args = parser.parse_args()
    
    enrich_dataset(
        input_csv=Path(args.input),
        output_csv=Path(args.output),
        model_name=args.model,
        batch_size=args.batch_size,
        use_label_llm=args.label_llm,
    )
