import logging
import urllib.parse
import time
import datetime
from typing import Dict, Any, Optional

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class WhoisEnricher:
    """Extracts WHOIS metadata from URLs using the who-dat API with in-memory caching."""
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.api_url = "https://who-dat.as93.net/v1/whois/{}"
        
        if not REQUESTS_AVAILABLE:
            logger.warning("WHOIS enrichment requested but 'requests' is not installed.")

    def get_whois_data(self, url: str) -> Optional[Dict[str, Any]]:
        if not REQUESTS_AVAILABLE:
            return None
            
        try:
            parsed_url = urllib.parse.urlparse(url)
            domain = parsed_url.netloc if parsed_url.netloc else parsed_url.path.split('/')[0]
            if not domain:
                return None
            
            # Remove any subdomains except the registrable domain. 
            # Note: who-dat handles extracting the registrable domain automatically,
            # but we strip ports if they exist.
            domain = domain.split(':')[0]
            
            if domain in self._cache:
                return self._cache[domain]

            time.sleep(0.5) # Slight delay to respect rate limits (30 req/min limit on public API)
            
            response = requests.get(self.api_url.format(domain), timeout=10)
            
            if response.status_code != 200:
                logger.debug(f"who-dat API returned status {response.status_code} for {domain}")
                return {"error": f"HTTP {response.status_code}"}
                
            data = response.json()
            
            if not data.get("isRegistered"):
                # Domain is not registered or lookup failed
                result = {
                    "domain": domain,
                    "age_days": -1,
                    "hidden": False,
                    "country": "Unknown",
                    "registrar": "Unknown"
                }
                self._cache[domain] = result
                return result

            # Parse age_days
            age_days = -1
            created_date_str = data.get("dates", {}).get("created")
            if created_date_str:
                try:
                    # Parse ISO 8601 (e.g. "1997-09-15T04:00:00Z")
                    c_date = datetime.datetime.fromisoformat(created_date_str.replace('Z', '+00:00'))
                    # Convert now() to timezone-aware UTC for subtraction
                    now = datetime.datetime.now(datetime.timezone.utc)
                    age_days = (now - c_date).days
                except (ValueError, TypeError):
                    pass
            
            # Parse registrar
            registrar = data.get("registrar", {}).get("name") or "Unknown"
            
            # Parse country
            country = "Unknown"
            registrant = data.get("contacts", {}).get("registrant") or {}
            address = registrant.get("address") or {}
            if address.get("country"):
                country = address.get("country")
                
            # Parse hidden/privacy status
            is_hidden = bool(registrant.get("redacted", False))
            
            # Fallback hidden check via keywords
            privacy_keywords = ["privacy", "redacted", "protect", "whoisguard", "proxy", "hidden", "statutory", "masked"]
            org = str(registrant.get("organization", "")).lower()
            reg_lower = registrar.lower()
            
            if not is_hidden:
                if any(k in org for k in privacy_keywords) or any(k in reg_lower for k in privacy_keywords):
                    is_hidden = True
                
            result = {
                "domain": domain,
                "age_days": age_days,
                "hidden": is_hidden,
                "country": country,
                "registrar": registrar
            }
            self._cache[domain] = result
            return result
            
        except requests.exceptions.RequestException as e:
            logger.debug(f"Network error extracting WHOIS for {url}: {e}")
            return {"error": "Network Error"}
        except Exception as e:
            logger.debug(f"Error extracting WHOIS for {url}: {e}")
            return {"error": str(e)}

if __name__ == "__main__":
    enricher = WhoisEnricher()
    urls = [
        "https://google.com/search",
        "http://amazon.co.uk",
        "https://who-dat.as93.net",
        "http://this-domain-surely-does-not-exist-12345.com"
    ]
    
    for u in urls:
        print(f"Testing URL: {u}")
        print(enricher.get_whois_data(u))
        print("-" * 50)
