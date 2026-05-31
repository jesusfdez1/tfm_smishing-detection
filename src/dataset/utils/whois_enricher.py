import logging
import urllib.parse
import time
from typing import Dict, Any, Optional

try:
    import whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False

logger = logging.getLogger(__name__)

class WhoisEnricher:
    """Extracts WHOIS metadata from URLs with in-memory caching to respect rate limits."""
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        
        if not WHOIS_AVAILABLE:
            logger.warning("WHOIS enrichment requested but 'python-whois' is not installed.")

    def get_whois_data(self, url: str) -> Optional[Dict[str, Any]]:
        if not WHOIS_AVAILABLE:
            return None
            
        try:
            parsed_url = urllib.parse.urlparse(url)
            domain = parsed_url.netloc if parsed_url.netloc else parsed_url.path.split('/')[0]
            if not domain:
                return None
            
            if domain in self._cache:
                return self._cache[domain]

            time.sleep(0.5) # Slight delay to respect rate limits on public servers
            
            w = whois.whois(domain)
            result = {
                "domain": domain,
                "creation_date": str(w.creation_date),
                "expiration_date": str(w.expiration_date),
                "country": w.country,
                "org": w.org,
                "registrar": w.registrar
            }
            self._cache[domain] = result
            return result
        except Exception as e:
            logger.debug(f"Error extracting WHOIS for {url}: {e}")
            return None
