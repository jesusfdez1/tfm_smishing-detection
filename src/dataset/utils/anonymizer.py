import re
import logging

from typing import Dict, Tuple, List, Any, Optional
from urllib.parse import unquote
import unicodedata



logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class SMSAnonymizer:
    """
    Anonymizes SMS text by masking PII and structured identifiers.
    Supports regex-based masking and an optional privacy-filter model.

    The output keeps a simple replacement-token scheme (e.g., <EMAIL>, <URL>)
    so downstream feature extraction and model inputs remain consistent.
    """
    # Map privacy-filter labels to replacement tokens used in the dataset.
    PRIVACY_LABEL_MAP = {
        "private_email": "<EMAIL>",
        "private_person": "<PERSON>",
        "private_phone": "<PHONE>",
        "private_url": "<URL>",
        "private_date": "<DATE>",
        "secret": "<SECRET>",
        "account_number": "<ACCOUNT_NUMBER>",
        "private_address": "<ADDRESS>",
    }

    def __init__(
        self,
        use_privacy_filter: bool = False,
        privacy_filter_model: str = "openai/privacy-filter",
    ):
        """Configure anonymization behavior and optional PII model.

        use_privacy_filter switches from regex masking to a token classifier.
        """
        self.use_privacy_filter = use_privacy_filter
        self.privacy_filter_model = privacy_filter_model
        self._privacy_pipe = None

        # Ordered from most specific to most generic to avoid partial matches.
        self.patterns = [
            # URLs - collapsed to <URL> to prevent dataset leakage from mixed pre-anonymized sources
            (r"(https?://(?:[a-zA-Z0-9-]{1,5}\.[a-zA-Z]{2,4})/[a-zA-Z0-9_-]{2,15}\b)", "<URL>"),
            (r"(https?://(?:\d{1,3}\.){3}\d{1,3}(?:\:\d+)?(?:/\S*)?)", "<URL>"),
            (r"(https?://\S+:\d{2,5}/\S*)", "<URL>"),
            (r"(https://\S+)", "<URL>"),
            (r"(http://\S+)", "<URL>"),
            (r"([a-zA-Z0-9+-.]+://\S+)", "<URL>"),

            # Crypto & Technical Identifiers
            (r"(\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b|\bbc1[a-zA-HJ-NP-Z0-9]{39,59}\b)", "<CRYPTO_ADDRESS>"),
            (r"(\b0x[a-fA-F0-9]{40}\b)", "<CRYPTO_ADDRESS>"),
            (r"(\b(?:\d{1,3}\.){3}\d{1,3}\b)", "<IP_ADDRESS>"),
            (r"(\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b)", "<UUID>", re.I),
            (r"(\b(?:\d{4}[ -]?){3}\d{4}\b)", "<CREDIT_CARD>"),
            (r"(\b(?=[A-Z]*[0-9])(?=[0-9]*[A-Z])[A-Z0-9]{8,25}\b)", "<TRACKING_ID>", re.I), 


            # Phones & Codes
            (r"(\+?\d{1,4}?[\s.-]?\(?\d{1,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}[\s.-]?\d{0,4})", "<PHONE>"),
            (r"(\b\d{5,6}\b)", "<PHONE_SHORTCODE>"), 

            # Identifiers
            (r"(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)", "<EMAIL>"),
            (r"(\b[A-Z]{2}\d{2}\s?(?:\w{4}\s?){3,5}\w{1,3}\b)", "<IBAN>", re.I),
        ]
        
        self.stats = {"total": 0, "replaced": 0, "fallback": 0}
        if self.use_privacy_filter:
            self._init_privacy_filter()

    def _init_privacy_filter(self) -> None:
        """Initialize the privacy-filter pipeline if available.

        If initialization fails, the anonymizer falls back to regex patterns.
        """
        try:
            from transformers import pipeline
            import torch

            device = 0 if torch.cuda.is_available() else -1
            
            # Token classification pipeline for privacy span detection.
            self._privacy_pipe = pipeline(
                "token-classification",
                model=self.privacy_filter_model,
                aggregation_strategy="simple",
                device=device,
            )
            # Add padding token if missing (required for batch processing)
            if self._privacy_pipe.tokenizer.pad_token is None:
                self._privacy_pipe.tokenizer.pad_token = self._privacy_pipe.tokenizer.eos_token
                
            logger.info("Privacy filter pipeline initialized (model=%s, device=%s)", 
                        self.privacy_filter_model, device)
        except Exception as e:
            logger.warning("Failed to initialize privacy filter, using regex fallback: %s", e)
            self._privacy_pipe = None

    def _process_with_privacy_filter_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Mask PII spans using the privacy-filter model output for a batch of texts.
        
        Returns a list of extracted entities for each text.
        """
        if not self._privacy_pipe:
            raise RuntimeError("Privacy pipe not initialized")
            
        self.stats["total"] += len(texts)
        try:
            # HuggingFace pipeline emits a false positive warning if __call__ is invoked >10 times,
            # even if we are passing batches. Reset call_count to suppress it.
            if hasattr(self._privacy_pipe, "call_count"):
                self._privacy_pipe.call_count = 0
                
            # Batch inference
            return self._privacy_pipe(texts, batch_size=32)
        except Exception as e:
            logger.warning("Privacy pipe failed for batch (size=%d), falling back: %s", len(texts), e)
            self.stats["fallback"] += len(texts)
            # Temporarily disable pipe to prevent infinite recursion and run regex fallback
            pipe_backup = self._privacy_pipe
            self._privacy_pipe = None
            # Return empty entities so it falls back to regex for all
            self._privacy_pipe = pipe_backup
            return [[] for _ in texts]

    @staticmethod
    def _normalize_privacy_label(label: str) -> Optional[str]:
        """Normalize BIOES labels to their base category.

        Example: B-private_email -> private_email.
        """
        if not label:
            return None
        cleaned = label.strip().lower()
        if cleaned.startswith("b-") or cleaned.startswith("i-") or cleaned.startswith("e-") or cleaned.startswith("s-"):
            cleaned = cleaned[2:]
        return cleaned


    def _decode_urls(self, text: str) -> str:
        """Decode percent-encoded URLs before pattern matching.

        This reduces false negatives when URLs are URL-encoded.
        """
        return unquote(text)

    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode to reduce obfuscation variants.

        Uses NFKC to collapse compatibility characters and confusables.
        """
        return unicodedata.normalize("NFKC", text)


    def process_message(self, text: str) -> Dict[str, Any]:
        """Return anonymized text and extracted entities for a single message.
        
        This is a convenience wrapper around process_messages for a single string.
        """
        return self.process_messages([text])[0]

    def process_messages(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Return anonymized text and extracted entities for a batch of messages.

        The output includes:
        - original: normalized text used for matching
        - anonymized: text with replacement tokens
        """
        processed_texts = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            text = self._decode_urls(text)
            text = self._normalize_unicode(text)
            text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
            processed_texts.append(text)

        entities_batch = [[] for _ in texts]

        if self.use_privacy_filter and self._privacy_pipe:
            entities_batch = self._process_with_privacy_filter_batch(processed_texts)

        results = []
        for i, text in enumerate(processed_texts):
            self.stats["total"] += 1
            
            all_spans: List[Tuple[int, int, str, int]] = []
            
            # 1. Collect regex spans on the ORIGINAL text (Priority 1)
            for pattern, replacement, *flags in self.patterns:
                flag = flags[0] if flags else 0
                for match in re.finditer(pattern, text, flag):
                    start, end = match.start(1), match.end(1)
                    all_spans.append((start, end, replacement, 1))
                    
            # 2. Collect AI spans (Priority 0)
            if self.use_privacy_filter and self._privacy_pipe:
                ai_output = entities_batch[i]
                for entity in ai_output:
                    label = entity["entity_group"]
                    normalized = self._normalize_privacy_label(label)
                    token = self.PRIVACY_LABEL_MAP.get(normalized)
                    if token:
                        all_spans.append((entity["start"], entity["end"], token, 0))

            # 3. Resolve overlaps: prioritize Regex (priority 1) over AI (priority 0), then longest span
            all_spans.sort(key=lambda x: (x[3], (x[1] - x[0]), -x[0]), reverse=True)
            
            final_spans = []
            for start, end, replacement, priority in all_spans:
                overlap = False
                for s, e, _, _ in final_spans:
                    if max(start, s) < min(end, e):
                        overlap = True
                        break
                if not overlap:
                    final_spans.append((start, end, replacement, priority))
                    
            # 4. Apply replacements from right to left
            final_spans.sort(key=lambda x: x[0], reverse=True)
            
            anon_text = text
            for start, end, replacement, priority in final_spans:
                self.stats["replaced"] += 1
                anon_text = anon_text[:start] + replacement + anon_text[end:]

            # Deduplicate consecutive identical tags (e.g. <URL><URL> or <PHONE> <PHONE>)
            anon_text = re.sub(r"(<[a-zA-Z0-9_]+>)(?:\s*\1)+", r"\1", anon_text)

            if len(final_spans) == 0:
                self.stats["fallback"] += 1

            results.append({
                "original": text,
                "anonymized": anon_text,
                "entities": entities_batch[i] if (self.use_privacy_filter and self._privacy_pipe) else []
            })

        return results

        
    def anonymize(self, text: str) -> str:
        """Convenience wrapper that returns only the anonymized text."""
        return self.process_message(text)["anonymized"]

    def get_coverage(self) -> Dict[str, float]:
        """Return aggregate masking statistics for the current run.

        fallback_rate indicates how often no entity was detected.
        """
        total = max(self.stats["total"], 1)
        return {
            "messages_processed": self.stats["total"],
            "entities_replaced": self.stats["replaced"],
            "fallback_rate": self.stats["fallback"] / total,
            "avg_entities_per_msg": self.stats["replaced"] / total
        }

if __name__ == "__main__":
    extractor = SMSAnonymizer(use_privacy_filter=False)
    
    sample_texts = [
        "Notice: Your package 1Z9999999999999999 could not be delivered. Update at http://x.co/ab3x",
        "Charge not authorized. If not yours, cancel at https://bank.xyz.online",
        "Contact us via whatsapp://send?phone=34600000000 or pay with 4532-1234-5678-9012",
        "Verification token: 550e8400-e29b-41d4-a716-446655440000. Claim at 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
    ]
    
    for txt in sample_texts:
        result = extractor.process_message(txt)
        print(f"Original:   {result['original']}")
        print(f"Anonymized: {result['anonymized']}")
        print(f"Entities:   {result['entities']}")
        print("-" * 80)
