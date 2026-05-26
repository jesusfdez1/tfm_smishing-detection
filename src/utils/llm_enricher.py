import os
import json
import logging
import datetime
import requests
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class LLMMetadataAnnotator:
    """
    Enriches SMS messages by inferring Language, Theme, and Urgency Level
    using Google Gemini API (gemini-2.5-flash) via REST.
    Returns structured JSON fitting the MetaSMS-HSS schema.
    """
    
    PROMPT_VERSION = "v1.0"
    MODEL_NAME = "gemini-2.5-flash"
    
    # Generic mapping based loosely on standard datasets (0-12)
    THEMES = {
        0: "Unknown/Other",
        1: "Banking/Finance",
        2: "Delivery/Logistics",
        3: "Government/Taxes",
        4: "Telecom/ISP",
        5: "E-commerce/Retail",
        6: "Social Media/Accounts",
        7: "Streaming/Entertainment",
        8: "Prizes/Lottery",
        9: "Job Offers",
        10: "Crypto",
        11: "Healthcare/Medical",
        12: "Family Emergency/Scam"
    }

    def __init__(self, use_mock: bool = False):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.use_mock = use_mock
        
        if not self.api_key and not self.use_mock:
            logger.warning("GEMINI_API_KEY not found. LLM enrichment will run in MOCK mode.")
            self.use_mock = True

    def _get_system_prompt(self) -> str:
        themes_str = "\n".join([f"{k}: {v}" for k, v in self.THEMES.items()])
        return f"""You are an expert cybersecurity analyst annotating SMS messages for a machine learning dataset.
Analyze the following SMS text and return a strict JSON object with EXACTLY these keys:
- "language": string, the ISO 639-1 two-letter code for the language (e.g., "es", "en", "fr", "de").
- "language_confidence": float between 0.0 and 1.0 representing your confidence in the language detection.
- "theme": integer from 0 to 12 representing the main topic.
- "urgency_level": integer representing urgency (0=Low/None, 1=Medium/Call to action, 2=High/Immediate threat or account locked).

Themes mapping:
{themes_str}

Respond ONLY with valid JSON. No markdown, no explanations."""

    def _mock_enrich(self, text: str) -> Dict[str, Any]:
        """Fallback mock enrichment for testing without an API key."""
        text_lower = text.lower()
        theme = 0
        urgency = 0
        
        # Simple heuristics for mock
        if "paquete" in text_lower or "package" in text_lower or "delivery" in text_lower:
            theme = 2
        elif "banco" in text_lower or "bank" in text_lower or "pago" in text_lower or "eur" in text_lower:
            theme = 1
            
        if "urgente" in text_lower or "retenido" in text_lower or "bloqueada" in text_lower or "unauthorized" in text_lower:
            urgency = 2
        elif "actualice" in text_lower or "verify" in text_lower or "update" in text_lower:
            urgency = 1
            
        return {
            "language": "es" if " el " in text_lower or " la " in text_lower or "su " in text_lower else "en",
            "language_confidence": 0.85,
            "theme": theme,
            "urgency_level": urgency
        }

    def annotate(self, text: str) -> Dict[str, Any]:
        """
        Main method to annotate text.
        Returns the enriched fields conforming to the MetaSMS-HSS schema.
        """
        # Default empty structure in case of failure
        result = {
            "llm_annotated": 0,
            "llm_model": None,
            "prompt_version": None,
            "llm_annotation_date": None,
            "language": "unknown",
            "language_confidence": 0.0,
            "theme": 0,
            "urgency_level": 0
        }
        
        if not text or len(text.strip()) == 0:
            return result

        llm_response = None
        
        if self.use_mock:
            llm_response = self._mock_enrich(text)
        else:
            try:
                # Direct REST call to Gemini 2.5 Flash
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.MODEL_NAME}:generateContent?key={self.api_key}"
                
                payload = {
                    "system_instruction": {
                        "parts": [{"text": self._get_system_prompt()}]
                    },
                    "contents": [{
                        "parts": [{"text": f"SMS to analyze: {text}"}]
                    }],
                    "generationConfig": {
                        "temperature": 0.0,
                        "response_mime_type": "application/json"
                    }
                }
                
                response = requests.post(url, json=payload, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    response_text = data['candidates'][0]['content']['parts'][0]['text']
                    llm_response = json.loads(response_text)
                else:
                    logger.error(f"Gemini API Error {response.status_code}: {response.text}")
                    llm_response = self._mock_enrich(text) # Fallback to mock
                    
            except Exception as e:
                logger.error(f"LLM Annotation Failed: {str(e)}")
                llm_response = self._mock_enrich(text) # Fallback to mock

        if llm_response:
            result.update({
                "llm_annotated": 1,
                "llm_model": self.MODEL_NAME if not self.use_mock else "mock-heuristic",
                "prompt_version": self.PROMPT_VERSION,
                "llm_annotation_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "language": llm_response.get("language", "unknown")[:2].lower(),
                "language_confidence": float(llm_response.get("language_confidence", 0.0)),
                "theme": int(llm_response.get("theme", 0)),
                "urgency_level": int(llm_response.get("urgency_level", 0))
            })
            
        return result

if __name__ == "__main__":
    # Test script in mock mode
    annotator = LLMMetadataAnnotator(use_mock=True)
    res = annotator.annotate("Correos: Su paquete esta retenido. Pague aduanas en <URL_HTTPS>")
    print(json.dumps(res, indent=2))
