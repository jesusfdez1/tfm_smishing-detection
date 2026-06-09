import math
import re
import urllib.parse
from collections import Counter
from typing import Dict, Any, List

class URLFeatureExtractor:
    """
    Extracts lexical and structural features from URLs.

    The goal is to preserve risk signal that may be lost after anonymization,
    such as suspicious TLDs, excessive subdomains, and high-entropy strings.
    """
    
    # Highly correlated TLDs with spam/phishing (Cheap or heavily abused)
    # Based on general threat intelligence (Spamhaus, APWG)
    SUSPICIOUS_TLDS = {
        'xyz', 'top', 'tk', 'ml', 'ga', 'cf', 'gq', 'loan', 'win', 
        'click', 'site', 'vip', 'cam', 'cn', 'ru', 'su', 'pw', 'cc',
        'icu', 'wang', 'club', 'date', 'trade', 'review', 'stream',
        'party', 'science', 'work', 'biz', 'info', 'pro', 'mobi',
        'online', 'tech', 'store', 'shop', 'space', 'monster', 
        'quest', 'buzz', 'surf', 'pics', 'rest', 'mw', 'ke', 'ws', 'to'
    }

    @staticmethod
    def _shannon_entropy(string: str) -> float:
        """Calculate Shannon entropy as a proxy for randomness/obfuscation."""
        if not string:
            return 0.0
        counts = Counter(string)
        probs = [count / len(string) for count in counts.values()]
        return -sum(p * math.log2(p) for p in probs)

    def extract_features(self, url: str) -> Dict[str, float]:
        """
        Extract structural features from a single URL.

        Returns a dictionary of float values so it can be used directly
        in downstream ML pipelines or saved to CSV.
        """
        features = {
            "url_length": 0.0,
            "url_entropy": 0.0,
            "num_dots": 0.0,
            "num_subdomains": 0.0,
            "is_suspicious_tld": 0.0,
            "has_ip_in_domain": 0.0
        }
        
        if not url or not isinstance(url, str):
            return features
            
        features["url_length"] = float(len(url))
        features["url_entropy"] = self._shannon_entropy(url)
        features["num_dots"] = float(url.count('.'))
        
        try:
            parsed = urllib.parse.urlparse(url)
            # If no scheme is present, netloc might be empty and domain is in path
            domain = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]
            
            # Check TLD
            parts = domain.split('.')
            if len(parts) > 1:
                tld = parts[-1].lower()
                if tld in self.SUSPICIOUS_TLDS:
                    features["is_suspicious_tld"] = 1.0
                
                # Subdomains (excluding www and tld and main domain)
                # E.g., a.b.example.com -> parts = ['a', 'b', 'example', 'com'] -> 4 parts. Subdomains = 4 - 2 = 2
                num_subdomains = len(parts) - 2
                if parts[0].lower() == 'www':
                    num_subdomains -= 1
                features["num_subdomains"] = float(max(0, num_subdomains))
                
            # Check if domain is an IP address
            if re.match(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$", domain):
                features["has_ip_in_domain"] = 1.0
                
        except Exception:
            pass # Malformed URLs will just have the basic length/entropy/dots
            
        return features

    def aggregate_features(self, urls: List[str]) -> Dict[str, float]:
        """
        Aggregate features for a list of URLs in one message.

        Uses max for risk indicators so a single suspicious URL dominates,
        and keeps a count of the total URLs detected.
        """
        base_features = {
            "max_url_length": 0.0,
            "max_url_entropy": 0.0,
            "max_num_dots": 0.0,
            "max_num_subdomains": 0.0,
            "has_suspicious_tld": 0.0,
            "has_ip_in_domain": 0.0,
            "url_count": float(len(urls))
        }
        
        if not urls:
            return base_features
            
        all_feats = [self.extract_features(url) for url in urls]
        
        base_features["max_url_length"] = max(f["url_length"] for f in all_feats)
        base_features["max_url_entropy"] = max(f["url_entropy"] for f in all_feats)
        base_features["max_num_dots"] = max(f["num_dots"] for f in all_feats)
        base_features["max_num_subdomains"] = max(f["num_subdomains"] for f in all_feats)
        base_features["has_suspicious_tld"] = max(f["is_suspicious_tld"] for f in all_feats)
        base_features["has_ip_in_domain"] = max(f["has_ip_in_domain"] for f in all_feats)
        
        return base_features

if __name__ == "__main__":
    extractor = URLFeatureExtractor()
    urls = [
        "https://amazon.es/track",
        "http://amzn-track.security-update.xyz/login",
        "http://192.168.1.1/payload",
        "https://bit.ly/3xY8a",
        "http://banco-santander.es.verify-account.tk"
    ]
    
    for u in urls:
        print(f"URL: {u}")
        feats = extractor.extract_features(u)
        for k, v in feats.items():
            print(f"  {k}: {v}")
        print()
