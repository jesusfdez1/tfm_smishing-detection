import pandas as pd
import logging
from pathlib import Path
from utils.cleaner import DatasetCleaner
from utils.anonymizer import SMSAnonymizer
from utils.feature_extractor import URLFeatureExtractor
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """
    Unified multimodal pipeline to preprocess raw SMS datasets.
    Generates cleaned/anonymized text AND numerical features for ML models.
    """
    def __init__(self, use_whois: bool = False):
        self.cleaner = DatasetCleaner()
        self.anonymizer = SMSAnonymizer(use_whois=use_whois)
        self.feature_extractor = URLFeatureExtractor()

    def process_row(self, text: str) -> pd.Series:
        """
        Applies the pipeline to a single string and returns a Series containing
        the processed text and all extracted features.
        """
        # Default output
        output = {
            "processed_text": "",
            "max_url_length": 0.0,
            "max_url_entropy": 0.0,
            "max_num_dots": 0.0,
            "max_num_subdomains": 0.0,
            "has_suspicious_tld": 0.0,
            "has_ip_in_domain": 0.0,
            "url_count": 0.0
        }
        
        if pd.isna(text) or not isinstance(text, str):
            return pd.Series(output)
            
        # 1. Clean and Tag Obfuscation
        cleaned_result = self.cleaner.clean(text)
        
        # 2. Anonymize PII from the cleaned text
        anonymized_result = self.anonymizer.process_message(cleaned_result["cleaned"])
        
        # 3. Extract Features BEFORE the URLs are lost forever
        # Find all URLs that were extracted during anonymization
        urls = [ent["value"] for ent in anonymized_result["entities"] if "URL" in ent["type"]]
        
        # Calculate features based on those URLs
        url_features = self.feature_extractor.aggregate_features(urls)
        
        # Combine everything
        output["processed_text"] = anonymized_result["anonymized"]
        output.update(url_features)
        
        # Include WHOIS features if they were extracted (optional)
        whois_age = 0.0
        for ent in anonymized_result["entities"]:
            if "whois" in ent and "creation_date" in ent["whois"]:
                # Very rough metric, in reality you'd parse datetime and calculate days diff
                whois_age = 1.0 
        if self.anonymizer.use_whois:
            output["has_whois_data"] = whois_age
            
        return pd.Series(output)

    def process_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """Applies the pipeline to an entire Pandas DataFrame."""
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")

        logger.info(f"Starting multimodal processing of {len(df)} messages...")
        
        # Apply the processing which returns a DataFrame of new columns
        new_columns_df = df[text_column].apply(self.process_row)
        
        # Concatenate the new columns with the original DataFrame
        result_df = pd.concat([df, new_columns_df], axis=1)
        
        logger.info("Processing complete.")
        logger.info(f"Anonymizer Stats: {self.anonymizer.get_coverage()}")
        logger.info(f"Cleaner Stats: {self.cleaner.stats}")
        
        return result_df

    def process_csv(self, input_path: str, output_path: str, text_column: str = "text"):
        """Loads a CSV, processes it, and saves the multimodal result."""
        input_file = Path(input_path)
        if not input_file.exists():
            logger.error(f"Input file not found: {input_path}")
            return

        logger.info(f"Loading dataset from {input_path}")
        df = pd.read_csv(input_path)
        
        processed_df = self.process_dataframe(df, text_column)
        
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Saved multimodal dataset to {output_path}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("Testing Multimodal Pipeline on dummy DataFrame...")
    
    # We test the "Text Parity" problem:
    # Two identical texts. One has a legit amazon link, the other a fake one.
    dummy_data = {
        "text": [
            "Tu paquete de Amazon ha llegado. Sigue tu envío aquí: https://amazon.es/track/123",
            "Tu paquete de Amazon ha llegado. Sigue tu envío aquí: http://amzn-track.security-update.xyz/login"
        ],
        "label": ["ham", "smishing"]
    }
    
    df = pd.DataFrame(dummy_data)
    
    pipeline = PreprocessingPipeline(use_whois=False)
    processed_df = pipeline.process_dataframe(df, text_column="text")
    
    for i, row in processed_df.iterrows():
        print(f"\nRow {i+1} ({row['label']}):")
        print(f"RAW TEXT:    {row['text']}")
        print(f"CLEAN TEXT:  {row['processed_text']}")
        print(f"URL ENTROPY: {row['max_url_entropy']:.2f}")
        print(f"SUBDOMAINS:  {row['max_num_subdomains']}")
        print(f"SUSP. TLD:   {row['has_suspicious_tld']}")
