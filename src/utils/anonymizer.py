import re
import logging
import urllib.parse
from typing import Dict, Tuple, List, Any
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
    Script to strictly anonymize raw SMS datasets.
    Replaces PII and structured entities with flat tags to prepare data for ML.
    """
    def __init__(self, use_whois: bool = False):
        self.use_whois = use_whois
        
        if self.use_whois and not WHOIS_AVAILABLE:
            logger.warning("WHOIS extraction requested but 'python-whois' is not installed.")

        # Ordered from most specific to most generic
        self.patterns = [
            # URLs
            # Structural Heuristic for Shorteners: Domain length <= 5 chars without TLD
            (r"(https?://(?:[a-zA-Z0-9-]{1,5}\.[a-zA-Z]{2,4})/[a-zA-Z0-9_-]{2,15}\b)", "<URL_SHORTENER>"),
            (r"(https?://(?:\d{1,3}\.){3}\d{1,3}(?:\:\d+)?(?:/\S*)?)", "<URL_IP>"),
            (r"(https?://\S+:\d{2,5}/\S*)", "<URL_PORT>"),
            (r"(https://\S+)", "<URL_HTTPS>"),
            (r"(http://\S+)", "<URL_HTTP>"),
            (r"([a-zA-Z0-9+-.]+://\S+)", "<CUSTOM_URI>"), # Deep links like whatsapp:// or tg://

            # Crypto & Technical Identifiers
            (r"(\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b|\bbc1[a-zA-HJ-NP-Z0-9]{39,59}\b)", "<CRYPTO_ADDRESS>"),
            (r"(\b0x[a-fA-F0-9]{40}\b)", "<CRYPTO_ADDRESS>"),
            (r"(\b(?:\d{1,3}\.){3}\d{1,3}\b)", "<IP_ADDRESS>"),
            (r"(\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b)", "<UUID>", re.I),
            (r"(\b(?:\d{4}[ -]?){3}\d{4}\b)", "<CREDIT_CARD>"),
            (r"(\b(?=[A-Z]*[0-9])(?=[0-9]*[A-Z])[A-Z0-9]{8,25}\b)", "<TRACKING_ID>", re.I), 


            # Phones & Codes
            (r"(\+?\d{1,4}?[\s.-]?\(?\d{1,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}[\s.-]?\d{0,4})", "<PHONE_NUMBER>"),
            (r"(\b\d{5,6}\b)", "<PHONE_SHORTCODE>"), 

            # Identifiers
            (r"(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)", "<EMAIL>"),
            (r"(\b[A-Z]{2}\d{2}\s?(?:\w{4}\s?){3,5}\w{1,3}\b)", "<IBAN>", re.I),
        ]
        
        self.stats = {"total": 0, "replaced": 0, "fallback": 0}

    def _decode_urls(self, text: str) -> str:
        return unquote(text)

    def _normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    def _get_whois_data(self, url: str) -> Dict[str, Any]:
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
        if not isinstance(text, str):
            text = str(text)
            
        self.stats["total"] += 1
        text = self._decode_urls(text)
        text = self._normalize_unicode(text)
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
        
        original_len = len(text)
        extracted_entities = []
        anonymized_text = text

        for pattern, replacement, *flags in self.patterns:
            flag = flags[0] if flags else 0
            matches = re.finditer(pattern, anonymized_text, flag)
            
            for match in matches:
                matched_str = match.group(1)
                
                entity_info = {
                    "type": replacement.strip("<>"),
                    "value": matched_str
                }
                
                if "URL" in replacement and self.use_whois:
                    w_data = self._get_whois_data(matched_str)
                    if w_data:
                        entity_info["whois"] = w_data
                
                extracted_entities.append(entity_info)
                self.stats["replaced"] += 1

                anonymized_text = anonymized_text.replace(matched_str, replacement)
            
        if len(anonymized_text) == original_len:
            self.stats["fallback"] += 1
            
        return {
            "original": text,
            "anonymized": anonymized_text,
            "entities": extracted_entities
        }
        
    def anonymize(self, text: str) -> str:
        return self.process_message(text)["anonymized"]

    def get_coverage(self) -> Dict[str, float]:
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
