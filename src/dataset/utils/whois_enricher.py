import logging
import urllib.parse
import time
import datetime
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
            
            # Calculate age in days
            age_days = -1
            if w.creation_date:
                # creation_date can be a list if multiple dates are returned
                c_date = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
                if isinstance(c_date, datetime.datetime):
                    age_days = (datetime.datetime.now() - c_date).days
            
            # Detect privacy/hidden registration
            is_hidden = False
            privacy_keywords = ["privacy", "redacted", "protect", "whoisguard", "proxy", "hidden", "statutory", "masked"]
            org = str(w.org).lower() if w.org else ""
            registrar = str(w.registrar).lower() if w.registrar else ""
            
            if any(k in org for k in privacy_keywords) or any(k in registrar for k in privacy_keywords):
                is_hidden = True
                
            result = {
                "domain": domain,
                "age_days": age_days,
                "hidden": is_hidden,
                "country": w.country or "Unknown",
                "registrar": w.registrar or "Unknown"
            }
            self._cache[domain] = result
            return result
        except Exception as e:
            logger.debug(f"Error extracting WHOIS for {url}: {e}")
            return {"error": str(e)}
