import os
import json
import logging
import datetime
import requests
from typing import Dict, Any
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class LLMMetadataAnnotator:
    """
    Enriches SMS messages by inferring language, theme, and urgency.

    Uses the Gemini REST API with optional API key rotation, and can fall back
    to a local heuristic when no key is available or requests fail.
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

    def __init__(self, use_mock: bool = False, model_name: str | None = None):
        """Initialize the annotator and load API keys from the environment."""
        load_dotenv()
        api_keys_env = os.environ.get("GEMINI_API_KEYS")
        if api_keys_env:
            self.api_keys = [key.strip() for key in api_keys_env.split(",") if key.strip()]
        else:
            single_key = os.environ.get("GEMINI_API_KEY")
            self.api_keys = [single_key] if single_key else []
        self.api_key_index = 0
        self.use_mock = use_mock
        self.model_name = model_name or self.MODEL_NAME
        
        if not self.api_keys and not self.use_mock:
            logger.warning("No GEMINI_API_KEY(S) found. LLM enrichment will run in MOCK mode.")
            self.use_mock = True

    def _iter_api_keys(self):
        """Yield API keys in a rotating order starting at the current index."""
        if not self.api_keys:
            return []
        total = len(self.api_keys)
        return [
            (idx, self.api_keys[idx])
            for idx in [(self.api_key_index + offset) % total for offset in range(total)]
        ]

    def _get_system_prompt(self) -> str:
        """Build the instruction prompt for structured JSON output."""
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
        """Fallback mock enrichment for testing without an API key.

        This uses simple keyword heuristics and is not intended for production.
        """
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
        Annotate text and return fields conforming to the MetaSMS-HSS schema.

        The method rotates API keys on rate-limit or transient errors and
        falls back to a heuristic if all attempts fail.
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

            for index, api_key in self._iter_api_keys():
                try:
                    url = (
                        f"https://generativelanguage.googleapis.com/v1beta/models/"
                        f"{self.model_name}:generateContent?key={api_key}"
                    )
                    response = requests.post(url, json=payload, timeout=10)

                    if response.status_code == 200:
                        data = response.json()
                        response_text = data['candidates'][0]['content']['parts'][0]['text']
                        llm_response = json.loads(response_text)
                        self.api_key_index = index
                        break

                    if response.status_code == 429 or response.status_code >= 500:
                        logger.warning(
                            "Gemini API rate/availability issue (%s). Rotating API key.",
                            response.status_code,
                        )
                        self.api_key_index = (index + 1) % len(self.api_keys)
                        continue

                    logger.error("Gemini API Error %s: %s", response.status_code, response.text)
                    break

                except Exception as e:
                    logger.warning("LLM request failed, rotating API key: %s", str(e))
                    if self.api_keys:
                        self.api_key_index = (index + 1) % len(self.api_keys)
                    continue

            if not llm_response:
                llm_response = self._mock_enrich(text)

        if llm_response:
            result.update({
                "llm_annotated": 1,
                "llm_model": self.model_name if not self.use_mock else f"mock-{self.model_name}",
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
