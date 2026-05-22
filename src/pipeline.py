import pandas as pd
import logging
from pathlib import Path
from utils.cleaner import DatasetCleaner
from utils.anonymizer import SMSAnonymizer
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """
    Unified pipeline to preprocess raw SMS datasets for Machine Learning.
    Phase 1: Forensics (Cleaner) - Detects obfuscation, tags it, and normalizes text.
    Phase 2: Anonymization - Replaces robust technical PII with structural tags.
    """
    def __init__(self, use_whois: bool = False):
        self.cleaner = DatasetCleaner()
        self.anonymizer = SMSAnonymizer(use_whois=use_whois)

    def process_text(self, text: str) -> str:
        """Applies the full pipeline to a single string."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
            
        # 1. Clean and Tag Obfuscation
        cleaned_result = self.cleaner.clean(text)
        
        # 2. Anonymize PII from the cleaned text
        anonymized_result = self.anonymizer.process_message(cleaned_result["cleaned"])
        
        return anonymized_result["anonymized"]

    def process_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """Applies the pipeline to an entire Pandas DataFrame."""
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")

        logger.info(f"Starting processing of {len(df)} messages...")
        
        # We apply the pipeline to the target column
        # Using a new column for the result is a good practice to avoid losing raw data
        df['processed_text'] = df[text_column].apply(self.process_text)
        
        logger.info("Processing complete.")
        
        # Log the coverage stats for analysis
        anon_stats = self.anonymizer.get_coverage()
        logger.info(f"Anonymizer Stats: {anon_stats}")
        logger.info(f"Cleaner Stats: {self.cleaner.stats}")
        
        return df

    def process_csv(self, input_path: str, output_path: str, text_column: str = "text"):
        """Loads a CSV, processes it, and saves the result."""
        input_file = Path(input_path)
        if not input_file.exists():
            logger.error(f"Input file not found: {input_path}")
            return

        logger.info(f"Loading dataset from {input_path}")
        df = pd.read_csv(input_path)
        
        processed_df = self.process_dataframe(df, text_column)
        
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed dataset to {output_path}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    # Example usage on a dummy dataframe
    print("Testing pipeline on dummy DataFrame...")
    
    dummy_data = {
        "text": [
            "cоrrеоs: Su paquete esta retenido.", # Homoglyphs
            "Charge not authorized. Cancel at https://bank.xyz.online",
            "C\u200Bo\u200Br\u200Br\u200Be\u200Bo\u200Bs: Verification pin: 849302" # ZWSP + Shortcode
        ],
        "label": ["smishing", "smishing", "smishing"]
    }
    
    df = pd.DataFrame(dummy_data)
    
    pipeline = PreprocessingPipeline(use_whois=False)
    processed_df = pipeline.process_dataframe(df, text_column="text")
    
    for i, row in processed_df.iterrows():
        print(f"\nRow {i+1}:")
        print(f"RAW:       {repr(row['text'])}")
        print(f"PROCESSED: {repr(row['processed_text'])}")

