import json
import csv
import argparse
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Iterable, Optional

def clean_str(value: Any) -> Optional[str]:
    """Clean string by removing surrounding whitespace and internal newlines."""
    if value is None or value == "":
        return None
    value = str(value)
    cleaned = value.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned if cleaned else None

def iso_from_epoch_ms(value: Any) -> Optional[str]:
    """Convert epoch milliseconds to an ISO timestamp string."""
    if value is None:
        return None
    try:
        ts = int(value)
        return datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError, OverflowError):
        return None

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
            downvotes = int(smish.get("downvotes") or 0)

            if upvotes > 0:
                original_label = "smishing (community verified)"
            elif downvotes > 0:
                original_label = "unlabeled (community rejected/downvoted)"
            else:
                original_label = "unlabeled (0 votes)"
            
            canonical_label = "" # Empty by default

            yield {
                "text": text,
                "canonical_label": canonical_label,
                "timestamp_original": timestamp_original or "",
                "source": "smishtank",
                "source_id": source_id or "",
                "source_url": source_url or "https://smishtank.com/",
                "original_label": original_label,
                "upvotes": str(upvotes),
                "downvotes": str(downvotes)
            }

def main():
    parser = argparse.ArgumentParser(description="Standalone parser for SmishTank data.")
    parser.add_argument("--input", type=str, required=True, help="Path to smishtank JSONL file")
    parser.add_argument("--output", type=str, default="data/processed/smishtank_standalone.csv", help="Output CSV path")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"Error: Input file {in_path} does not exist.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    records_processed = 0
    with out_path.open("w", encoding="utf-8", newline="") as out_f:
        writer = None
        
        for record in load_smish(in_path):
            if writer is None:
                writer = csv.DictWriter(out_f, fieldnames=list(record.keys()))
                writer.writeheader()
            
            writer.writerow(record)
            records_processed += 1
            
            if records_processed % 500 == 0:
                print(f"Processed {records_processed} records...")

    print(f"Done! Successfully extracted {records_processed} messages to {out_path}")

if __name__ == "__main__":
    main()
