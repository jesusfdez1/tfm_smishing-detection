import re
import logging
import urllib.parse
from typing import Dict, Tuple, List, Any, Optional
from urllib.parse import unquote
import unicodedata

try:
    import whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False

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
        use_whois: bool = False,
        use_privacy_filter: bool = False,
        privacy_filter_model: str = "openai/privacy-filter",
    ):
        """Configure anonymization behavior and optional PII model.

        use_whois enables WHOIS lookups for detected URLs.
        use_privacy_filter switches from regex masking to a token classifier.
        """
        self.use_whois = use_whois
        self.use_privacy_filter = use_privacy_filter
        self.privacy_filter_model = privacy_filter_model
        self._privacy_pipe = None
        
        if self.use_whois and not WHOIS_AVAILABLE:
            logger.warning("WHOIS extraction requested but 'python-whois' is not installed.")

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
        except Exception as exc:
            logger.warning("Privacy filter unavailable, falling back to regex: %s", exc)
            self.use_privacy_filter = False
            self._privacy_pipe = None

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

    def _process_with_privacy_filter(self, text: str) -> Dict[str, Any]:
        """Mask PII spans using the privacy-filter model output.

        Returns a dict with original text, anonymized text, and extracted entities.
        """
        if not self._privacy_pipe:
            return self.process_message(text)

        self.stats["total"] += 1
        output = self._privacy_pipe(text)

        # Collect replacement spans based on the model output.
        spans: List[Tuple[int, int, str, str]] = []
        for ent in output:
            start = ent.get("start")
            end = ent.get("end")
            label = ent.get("entity_group") or ent.get("entity")
            if start is None or end is None or not label:
                continue
            normalized = self._normalize_privacy_label(label)
            token = self.PRIVACY_LABEL_MAP.get(normalized)
            if not token:
                continue
            spans.append((int(start), int(end), token, normalized))

        if not spans:
            self.stats["fallback"] += 1
            return {"original": text, "anonymized": text, "entities": []}

        spans.sort(key=lambda x: x[0])
        
        # Merge overlapping or adjacent spans of the same token type
        merged_spans = []
        for start, end, token, normalized in spans:
            if not merged_spans:
                merged_spans.append([start, end, token, normalized])
            else:
                last_start, last_end, last_token, last_normalized = merged_spans[-1]
                # If they overlap, or are adjacent (only separated by whitespace) and have the same token
                if start <= last_end:
                    # Overlap: merge them
                    merged_spans[-1][1] = max(last_end, end)
                elif text[last_end:start].strip() == "" and token == last_token:
                    # Adjacent with same token: merge them
                    merged_spans[-1][1] = end
                else:
                    merged_spans.append([start, end, token, normalized])

        anonymized_text = text
        entities: List[Dict[str, Any]] = []

        for start, end, token, _ in reversed(merged_spans):
            value = text[start:end]
            
            # The Hugging Face pipeline often includes leading/trailing spaces in the entity span.
            # We want to preserve those spaces in the text so we don't accidentally glue words together.
            import re
            m_prefix = re.match(r"^(\s+)", value)
            if m_prefix:
                prefix_len = len(m_prefix.group(1))
                start += prefix_len
                value = value[prefix_len:]
                
            m_suffix = re.search(r"(\s+)$", value)
            if m_suffix:
                suffix_len = len(m_suffix.group(1))
                end -= suffix_len
                value = value[:-suffix_len]

            entity_type = token.strip("<>")
            entity_info: Dict[str, Any] = {"type": entity_type, "value": value}
            if "URL" in entity_type and self.use_whois:
                w_data = self._get_whois_data(value)
                if w_data:
                    entity_info["whois"] = w_data
            entities.append(entity_info)
            self.stats["replaced"] += 1
            anonymized_text = anonymized_text[:start] + token + anonymized_text[end:]

        entities.reverse()
        return {
            "original": text,
            "anonymized": anonymized_text,
            "entities": entities,
        }

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

    def _get_whois_data(self, url: str) -> Dict[str, Any]:
        """Fetch WHOIS metadata for a URL when enabled.

        Returns a small, flat dict so it can be serialized in the entities list.
        """
        if not WHOIS_AVAILABLE or not self.use_whois:
            return {}
        try:
            parsed_url = urllib.parse.urlparse(url)
            domain = parsed_url.netloc if parsed_url.netloc else parsed_url.path.split('/')[0]
            if not domain:
                return {}
            w = whois.whois(domain)
            return {
                "domain": domain,
                "creation_date": str(w.creation_date),
                "expiration_date": str(w.expiration_date),
                "country": w.country
            }
        except Exception as e:
            logger.debug(f"Error extracting WHOIS for {url}: {e}")
            return {"error": "WHOIS lookup failed"}

    def process_message(self, text: str) -> Dict[str, Any]:
        """Return anonymized text and extracted entities for a single message.

        The output includes:
        - original: normalized text used for matching
        - anonymized: text with replacement tokens
        - entities: list of extracted spans and types
        """
        if not isinstance(text, str):
            text = str(text)

        text = self._decode_urls(text)
        text = self._normalize_unicode(text)
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

        if self.use_privacy_filter and self._privacy_pipe:
            return self._process_with_privacy_filter(text)

        self.stats["total"] += 1

        # Collect all match spans from all patterns, then apply once in reverse
        # order so earlier replacements don't shift indices for later ones.
        # This also avoids str.replace replacing already-substituted tokens.
        all_spans: List[Tuple[int, int, str]] = []
        seen_spans: set = set()

        for pattern, replacement, *flags in self.patterns:
            flag = flags[0] if flags else 0
            for match in re.finditer(pattern, text, flag):
                start, end = match.start(1), match.end(1)
                # Skip overlapping spans already claimed by a higher-priority pattern
                if any(s < end and start < e for s, e, _ in seen_spans):
                    continue
                all_spans.append((start, end, replacement))
                seen_spans.add((start, end, replacement))

        # Sort by start position; apply in reverse to preserve indices
        all_spans.sort(key=lambda x: x[0])
        extracted_entities: List[Dict[str, Any]] = []
        anonymized_text = text

        for start, end, replacement in reversed(all_spans):
            matched_str = anonymized_text[start:end]
            entity_info: Dict[str, Any] = {
                "type": replacement.strip("<>"),
                "value": matched_str,
            }
            if "URL" in replacement and self.use_whois:
                w_data = self._get_whois_data(matched_str)
                if w_data:
                    entity_info["whois"] = w_data
            extracted_entities.append(entity_info)
            self.stats["replaced"] += 1
            anonymized_text = anonymized_text[:start] + replacement + anonymized_text[end:]

        # Restore chronological order (we built the list in reverse)
        extracted_entities.reverse()

        if len(all_spans) == 0:
            self.stats["fallback"] += 1

        return {
            "original": text,
            "anonymized": anonymized_text,
            "entities": extracted_entities,
        }

        
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
    extractor = SMSAnonymizer(use_whois=False)
    
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
