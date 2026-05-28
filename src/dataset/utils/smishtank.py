#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SmishTank Raw API Extractor - Academic ELT Pipeline
----------------------------------------------------
High-throughput raw extraction pipeline for SmishTank API responses.
Persists immutable payloads in JSONL format for downstream transformation.
Designed for academic research, Master's Thesis data collection, and 
reproducible ELT (Extract-Load-Transform) workflows.

Key Features:
- Large-batch parallel extraction (default: 500 IDs per chunk)
- Automatic and rapid dataset boundary detection
- Incremental resumption via O(1) JSONL tail inspection
- Multi-pass retry mechanism for transient API/network failures
- Dedicated failed-ID audit log for methodological transparency
- Zero external dependencies beyond 'requests'
- Formal academic logging, documentation, and citation compliance

Academic Use Notice:
If utilizing data obtained from SmishTank.com, citation of the original 
publication is mandatory. Citation instructions: https://smishtank.com/

Author: Academic Research Pipeline
License: Academic Use Only
"""

import os
import sys
import csv
import json
import time
import logging
import random
import requests
import concurrent.futures
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================
API_URL = "https://l0sry40g26.execute-api.us-west-1.amazonaws.com/prod/smish"
API_KEY = "wF6E9Pe3fjUeKALCFfvo9op3yF7dG5K3A4TU94Qf"

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "content-type": "application/json",
    "origin": "https://smishtank.com",
    "referer": "https://smishtank.com/",
    "user-agent": "Mozilla/5.0 (Academic Research Bot; Raw ELT Extraction Pipeline)",
    "x-api-key": API_KEY
}

OUTPUT_JSONL = "smishtank.jsonl"
FAILED_IDS_CSV = "smishtank_failed_ids.csv"
SOURCE_DATASET = "smishtank"
LICENSE_NOTE = "Academic use only. Citation required: https://smishtank.com/"

MAX_WORKERS = 10
CHUNK_SIZE = 500
MAX_CONSECUTIVE_EMPTY_IDS = 100  # Automatic termination threshold
INTERNAL_MAX_RETRIES = 3
BATCH_MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 1.5
RESUME_FROM_ID = 1

# =============================================================================
# LOGGING & SESSION CONFIGURATION
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

SESSION = requests.Session()
ADAPTER = requests.adapters.HTTPAdapter(
    pool_connections=MAX_WORKERS,
    pool_maxsize=MAX_WORKERS,
    max_retries=0
)
SESSION.mount("https://", ADAPTER)
SESSION.headers.update(HEADERS)

# =============================================================================
# CORE FUNCTIONS
# =============================================================================
def get_last_saved_id(filepath: str) -> int:
    """
    Reads the last line of an append-only JSONL file and extracts the 
    highest message_id. Enables strict incremental resumption without 
    scanning the entire dataset. Returns 0 if the file is missing or empty.
    """
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return 0
    
    try:
        with open(filepath, 'rb') as f:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
                if f.tell() == 0:
                    break
            last_line = f.readline().decode('utf-8')
        
        data = json.loads(last_line)
        return int(data.get("message_id", 0))
    except Exception as exc:
        logger.warning("Failed to parse last line of %s for resumption. Reason: %s", filepath, exc)
        return 0


def fetch_smish_record(msg_id: int) -> Dict[str, Any]:
    """
    Retrieves a single message record from the SmishTank API.
    Implements exponential backoff with jitter for rate limits and server errors.
    Returns a dictionary with explicit status differentiation:
      - 'success': Valid record retrieved (contains raw API payload)
      - 'not_found': Message ID does not exist in the database
      - 'error': Transient network or API failure (subject to batch-level retry)
    """
    payload = {"message": 0, "messageID": msg_id, "username": ""}
    
    for attempt in range(1, INTERNAL_MAX_RETRIES + 1):
        try:
            response = SESSION.post(API_URL, json=payload, timeout=12)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, str) and "does not exist" in data.lower():
                    return {"status": "not_found"}
                if isinstance(data, dict) and data.get("smish"):
                    return {"status": "success", "data": data}
                return {"status": "not_found"}
                
            elif response.status_code == 429:
                wait_time = (RETRY_BACKOFF_FACTOR ** attempt) + random.uniform(0.3, 1.2)
                logger.debug("Rate limit (HTTP 429) for ID %d. Backing off %.2fs.", msg_id, wait_time)
                time.sleep(wait_time)
                continue
                
            elif response.status_code >= 500:
                wait_time = (RETRY_BACKOFF_FACTOR ** attempt) + random.uniform(0.3, 1.2)
                logger.debug("Server error (HTTP %d) for ID %d. Retry %d/%d.", response.status_code, msg_id, attempt, INTERNAL_MAX_RETRIES)
                time.sleep(wait_time)
                continue
                
            else:
                return {"status": "error", "http_code": response.status_code}
                
        except requests.RequestException as exc:
            if attempt == INTERNAL_MAX_RETRIES:
                return {"status": "error", "exception": str(exc)}
            wait_time = (RETRY_BACKOFF_FACTOR ** attempt) + random.uniform(0.3, 1.2)
            time.sleep(wait_time)
            
    return {"status": "error", "exception": "Internal retry limit exceeded"}


def process_id_batch(id_batch: List[int]) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Executes parallel API requests for a batch of message IDs.
    Implements a multi-pass retry mechanism for transient errors.
    Returns a tuple of:
      1. Successfully retrieved raw records formatted for JSONL persistence
      2. Permanently failed IDs (exhausted all retry passes)
    """
    valid_records = []
    failed_ids = list(id_batch)
    retry_pass = 0

    while failed_ids and retry_pass < BATCH_MAX_RETRIES:
        if retry_pass > 0:
            cooldown = 3.0 * retry_pass
            logger.info("Retry pass %d/%d for %d failed IDs. Cooldown: %.1fs.", retry_pass, BATCH_MAX_RETRIES, len(failed_ids), cooldown)
            time.sleep(cooldown)

        current_batch = failed_ids
        failed_ids = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_id = {executor.submit(fetch_smish_record, mid): mid for mid in current_batch}
            
            for future in concurrent.futures.as_completed(future_to_id):
                mid = future_to_id[future]
                try:
                    result = future.result()
                    if result["status"] == "success":
                        valid_records.append({"message_id": mid, "raw": result["data"]})
                    elif result["status"] == "error":
                        failed_ids.append(mid)
                except Exception as exc:
                    logger.error("Unhandled exception for ID %d: %s", mid, exc)
                    failed_ids.append(mid)

        retry_pass += 1

    return sorted(valid_records, key=lambda x: x["message_id"]), failed_ids


def append_to_jsonl(records: List[Dict[str, Any]], filepath: str) -> None:
    """
    Appends raw records to a JSONL file incrementally.
    Each line contains a single valid JSON object: {"message_id": X, "raw": {...}}
    Ensures immutable, stream-friendly persistence for downstream transformation.
    """
    if not records:
        return
    with open(filepath, 'a', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')


def append_failed_ids(failed_ids: List[int], filepath: str) -> None:
    """
    Appends permanently failed message IDs to a dedicated audit CSV.
    Enables manual reconciliation or deferred reprocessing without 
    interrupting the primary extraction pipeline.
    """
    if not failed_ids:
        return
    file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    records = [{"message_id": mid, "failure_status": "PERMANENT_TRANSIENT_ERROR", "timestamp_utc": timestamp} for mid in failed_ids]
    
    with open(filepath, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["message_id", "failure_status", "timestamp_utc"],
            quoting=csv.QUOTE_ALL,
            lineterminator='\n'
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(records)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main() -> None:
    """
    Orchestrates the raw extraction pipeline. Handles auto-resumption, 
    large-batch parallel processing, multi-pass retries, failed-ID auditing, 
    automatic boundary detection, and graceful shutdown.
    """
    logger.info("Initializing SmishTank raw ELT extraction pipeline.")
    logger.info("Raw output: %s | Failed ID log: %s", OUTPUT_JSONL, FAILED_IDS_CSV)
    logger.info("Workers: %d | Chunk size: %d | Termination threshold: %d consecutive empty IDs", 
                MAX_WORKERS, CHUNK_SIZE, MAX_CONSECUTIVE_EMPTY_IDS)

    last_id = get_last_saved_id(OUTPUT_JSONL)
    start_id = max(last_id + 1, RESUME_FROM_ID)
    
    if last_id > 0:
        logger.info("Existing dataset detected. Resuming extraction from ID %d.", start_id)
    else:
        logger.info("No existing dataset found. Starting extraction from ID %d.", start_id)

    current_id = start_id
    consecutive_empty = 0
    total_saved = 0
    total_failed = 0

    try:
        while True:
            id_batch = list(range(current_id, current_id + CHUNK_SIZE))
            logger.info("Processing batch: IDs %d to %d", id_batch[0], id_batch[-1])
            
            batch_records, failed_ids = process_id_batch(id_batch)
            
            if batch_records:
                append_to_jsonl(batch_records, OUTPUT_JSONL)
                total_saved += len(batch_records)
                
                # Contamos cuántos IDs vacíos hay desde el último válido hasta el final del lote
                last_valid_id = batch_records[-1]["message_id"]
                consecutive_empty = id_batch[-1] - last_valid_id

                logger.info("Successfully persisted %d raw records. Cumulative total: %d", len(batch_records), total_saved)
            else:
                consecutive_empty += CHUNK_SIZE
                logger.info("Batch yielded no valid records. Consecutive empty IDs: %d/%d", consecutive_empty, MAX_CONSECUTIVE_EMPTY_IDS)
            
            if failed_ids:
                append_failed_ids(failed_ids, FAILED_IDS_CSV)
                total_failed += len(failed_ids)
                logger.warning("Logged %d permanently failed IDs to %s. Cumulative failures: %d", len(failed_ids), FAILED_IDS_CSV, total_failed)
                
            if consecutive_empty >= MAX_CONSECUTIVE_EMPTY_IDS:
                logger.info("Termination condition met. Dataset boundary detected after %d consecutive empty IDs.", consecutive_empty)
                break
                
            current_id += CHUNK_SIZE
            time.sleep(0.10 + random.uniform(0, 0.05))

    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user. All previously saved records remain intact in %s.", OUTPUT_JSONL)
    except Exception as exc:
        logger.error("Critical pipeline failure: %s", exc, exc_info=True)
    finally:
        logger.info("Pipeline execution completed.")
        logger.info("Total raw records persisted: %d | Total failed IDs logged: %d", total_saved, total_failed)
        logger.info("Academic compliance notice: Data sourced from SmishTank.com. Citation is mandatory: https://smishtank.com/")


if __name__ == "__main__":
    main()