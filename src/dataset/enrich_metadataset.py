"""Enrich the MetaSMS-HSS dataset using a Large Language Model.

This script reads a pre-consolidated CSV dataset, identifies rows needing
labeling or metadata enrichment, processes them in batches via the LLM API,
and writes the enriched results to a new CSV.
"""

import argparse
import csv
import datetime
import json
import logging
import re
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
    writer: csv.DictWriter,
) -> int:
    """Process a batch of rows, querying the LLM in one go, and write to CSV."""
    if not batch_buffer:
        return 0
    
    # Pre-process texts and determine which need LLM
    llm_texts = []
    indices_to_annotate = []
    
    for i, row in enumerate(batch_buffer):
        llm_text = row.get("text")
        llm_texts.append(llm_text)
        
        needs_theme = not row.get("theme") or str(row.get("theme")).strip() == ""
        
        # Only call LLM if we are missing the theme
        should_call_llm = needs_theme
        
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
        
        # Update metadata only if we actually queried the LLM for this row
        if i in indices_to_annotate:
            llm_primary = llm_results_map.get(i, {})
            row["llm_model"] = llm_primary.get("llm_model", "")
            row["prompt_version"] = llm_primary.get("prompt_version", "")
            row["llm_annotation_date"] = llm_primary.get("llm_annotation_date", "")
            row["theme"] = llm_primary.get("theme", "unknown")
            row["urgency_level"] = llm_primary.get("urgency_level", "none")
            
            # Use Qwen's robust language detection over FastText's if FastText was uncertain (< 60% confidence)
            llm_lang = llm_primary.get("language")
            if llm_lang and len(llm_lang) == 2 and llm_lang != "unknown":
                try:
                    ft_conf = float(row.get("language_confidence") or 0.0)
                except ValueError:
                    ft_conf = 0.0
                if ft_conf < 0.60 or row.get("language") == "unknown":
                    row["language"] = llm_lang
                    row["language_confidence"] = "LLM"
        
        # Clean up legacy entities column if it exists in the input
        row.pop("entities", None)
        
        # Add WHOIS enrichment ONLY if it hasn't been done yet
        if whois_enricher and not row.get("whois_domain"):
            row["whois_domain"] = ""
            row["whois_tld"] = ""
            row["whois_age_days"] = ""
            row["whois_hidden"] = ""
            row["whois_country"] = ""
            
            try:
                text = row.get("text", "")
                urls = re.findall(r"(?:https?://[^\s]+|[a-zA-Z0-9.-]+\.(?:com|org|net|info|biz|me|ly|co|us|uk|es|fr|ru)(?::\d+)?(?:/[^\s>]*)?)", text)
                
                if urls:
                    best_w_data = None
                    best_score = -9999
                    has_error = False
                    
                    for url in set(urls):
                        url = url.rstrip('.,;:"\'()')
                        w_data = whois_enricher.get_whois_data(url)
                        if w_data:
                            if "error" in w_data:
                                has_error = True
                                continue
                                
                            # Score: +10 if hidden. -age_days if known.
                            score = 10 if w_data.get("hidden") else 0
                            if w_data.get("age_days", -1) >= 0:
                                score -= w_data["age_days"]
                            else:
                                score -= 365
                                
                            if score > best_score:
                                best_score = score
                                best_w_data = w_data
                    
                    if best_w_data:
                        domain_str = best_w_data["domain"]
                        row["whois_domain"] = domain_str
                        row["whois_tld"] = domain_str.split('.')[-1] if '.' in domain_str else ""
                        row["whois_age_days"] = best_w_data["age_days"] if best_w_data["age_days"] >= 0 else ""
                        row["whois_hidden"] = 1 if best_w_data["hidden"] else 0
                        row["whois_country"] = best_w_data["country"]
                    elif has_error:
                        row["whois_domain"] = "ERROR"
                        row["whois_tld"] = "ERROR"
            except Exception as e:
                logger.debug(f"Failed to process WHOIS for row: {e}")

        writer.writerow(row)
        written_count += 1
        
    return written_count


def save_resume_state(state_path: Path, state: dict) -> None:
    """Persist resume state using a temp file for atomic writes."""
    tmp_path = state_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=True, indent=2)
    for attempt in range(5):
        try:
            tmp_path.replace(state_path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.5)

def enrich_dataset(
    input_csv: Path,
    output_csv: Path,
    model_name: str,
    batch_size: int,
    resume: bool = False,
) -> None:
    """Read un-annotated dataset and run batch LLM and WHOIS enrichment."""
    if not input_csv.exists():
        logger.error("Input file not found: %s", input_csv)
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    annotator = LLMMetadataAnnotator(model_name=model_name)
    whois_enricher = WhoisEnricher()
    
    total_written = 0
    batch_buffer = []
    
    processed_ids = set()
    mode = "w"
    
    # Check if we should skip ERROR from processed_ids
    # This ensures ERROR states are re-tried
    if resume and output_csv.exists():
        mode = "a"
        try:
            with output_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    msg_id = row.get("message_id")
                    if msg_id and row.get("whois_domain") != "ERROR" and row.get("theme") not in ["ERROR_PARSING", "ERROR_INFERENCE"]:
                        processed_ids.add(msg_id)
            logger.info("Resuming: found %d already processed records in %s", len(processed_ids), output_csv)
        except Exception as e:
            logger.error("Failed to read output CSV for resuming: %s", e)
            mode = "w"

    state_path = output_csv.with_suffix(".state.json")
    state = {
        "split": output_csv.name,
        "total_processed": len(processed_ids) if resume else 0,
        "last_updated": ""
    }

    try:
        with input_csv.open("r", encoding="utf-8") as in_f, \
             output_csv.open(mode, encoding="utf-8", newline="") as out_f:
            
            reader = csv.DictReader(in_f)
            writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames)
            
            if mode == "w":
                writer.writeheader()

            for row in reader:
                msg_id = row.get("message_id")
                if resume and msg_id in processed_ids:
                    continue

                batch_buffer.append(row)
                if len(batch_buffer) >= batch_size:
                    written_in_batch = process_batch(
                        batch_buffer, annotator, whois_enricher, writer
                    )
                    total_written += written_in_batch
                    batch_buffer.clear()
                    
                    state["total_processed"] += written_in_batch
                    state["last_updated"] = datetime.datetime.now().isoformat()
                    save_resume_state(state_path, state)
                    out_f.flush()
                    logger.info("Progress state:\n%s", json.dumps(state, indent=2))
                    
            # Process remaining
            if batch_buffer:
                written_in_batch = process_batch(
                    batch_buffer, annotator, whois_enricher, writer
                )
                total_written += written_in_batch
                
                state["total_processed"] += written_in_batch
                state["last_updated"] = datetime.datetime.now().isoformat()
                save_resume_state(state_path, state)
                out_f.flush()
                logger.info("Progress state:\n%s", json.dumps(state, indent=2))
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Total written this session: %d", total_written)
    except Exception as e:
        logger.error("Enrichment failed: %s", e)
        
    logger.info("LLM Enrichment completed successfully. Total records: %d", total_written)
    logger.info("Saved to: %s", output_csv)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    parser = argparse.ArgumentParser(description="Enrich MetaSMS-HSS dataset using LLM")
    parser.add_argument("--input", type=str, required=True, help="Path to input unlabeled CSV")
    parser.add_argument("--output", type=str, required=True, help="Path to save enriched output CSV")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4", help="Local Hugging Face model name to use.")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of rows to process in one LLM call")
    parser.add_argument("--resume", action="store_true", help="Resume from previous partial run by skipping already processed rows in output CSV")
    
    args = parser.parse_args()
    
    enrich_dataset(
        input_csv=Path(args.input),
        output_csv=Path(args.output),
        model_name=args.model,
        batch_size=args.batch_size,
        resume=args.resume,
    )
