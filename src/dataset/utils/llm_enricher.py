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
    
    PROMPT_VERSION = "v3.0"
    MODEL_NAME = "gemini-2.5-flash"
    
    # Taxonomía cerrada de temáticas
    THEMES = [
        "personal",
        "banking",
        "delivery",
        "account_security",
        "promotion",
        "government",
        "job_offer",
        "dating_adult",
        "subscription",
        "service_alert",
        "unknown"
    ]
    
    # Taxonomía cerrada de urgencia
    URGENCY_LEVELS = [
        "none",
        "low",
        "medium",
        "high"
    ]

    def __init__(self, use_mock: bool = False, model_name: str | None = None):
        """Initialize the annotator and load API keys from the environment."""
        self.use_mock = use_mock
        self.provider = "gemini"
        self.model_name = model_name or self.MODEL_NAME
        self.api_keys = []
        self.api_key_index = 0

        if not self.use_mock:
            load_dotenv()
            
            # Auto-detect provider based on model name or default to gemini
            if self.model_name.startswith("llama") or "groq" in self.model_name.lower():
                self.provider = "groq"
            
            if self.provider == "groq":
                api_keys_env = os.environ.get("GROQ_API_KEYS")
                if api_keys_env:
                    self.api_keys = [key.strip() for key in api_keys_env.split(",") if key.strip()]
                else:
                    single_key = os.environ.get("GROQ_API_KEY")
                    self.api_keys = [single_key] if single_key else []
                
                if not self.api_keys:
                    raise ValueError("No GROQ_API_KEY(S) found. Aborting.")
            else:
                api_keys_env = os.environ.get("GEMINI_API_KEYS")
                if api_keys_env:
                    self.api_keys = [key.strip() for key in api_keys_env.split(",") if key.strip()]
                else:
                    single_key = os.environ.get("GEMINI_API_KEY")
                    self.api_keys = [single_key] if single_key else []

                if not self.api_keys:
                    raise ValueError("No GEMINI_API_KEY(S) found. Aborting.")

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
        """Build the instruction prompt for structured JSON output using Few-Shot + CoT."""
        themes_str = "\n".join([f"- {t}" for t in self.THEMES])
        urgency_str = "\n".join([f"- {u}" for u in self.URGENCY_LEVELS])
        
        return f"""You are an expert cybersecurity analyst annotating SMS messages for a machine learning dataset.
You will receive a JSON array of text messages. The input messages may be in multiple languages (Spanish, English, French, etc.). Analyze the semantic meaning in its native language, but ALWAYS write your "reasoning" and JSON keys in English.

You must return a strict JSON array of objects, one for each input message, in the EXACT SAME ORDER.

For each object, you must include EXACTLY these keys:
- "reasoning": string, a brief 1-2 sentence explanation of your analysis in English.
- "theme": string, strictly one of the themes listed below.
- "urgency_level": string, strictly one of the urgency levels listed below.

Themes mapping (Strictly MECE):
- personal: Casual conversation, greetings, family matters, or personal questions.
- banking: Banks, wire transfers, credit/debit cards, crypto, or specific financial alerts.
- delivery: Packages, post office, customs fees, shipping tracking (e.g. Correos, DHL).
- account_security: Account locks, suspicious logins, PIN/OTP codes, or identity verification (even if related to a bank, if the focus is the account login/security, use this).
- promotion: Prizes won, lotteries, discounts, aggressive commercial offers, or gifts.
- government: Traffic fines (DGT), taxes (Hacienda), public administration notifications.
- job_offer: Recruitment, job offers, work from home opportunities, easy money scams.
- dating_adult: Dating, sexual content, "hot singles", adult contacts.
- subscription: Premium SMS services, horoscopes, ringtones, paid subscriptions.
- service_alert: Medical appointments, utility bills/outages, mobile carrier service reminders.
- unknown: Completely unreadable, lacks context, or does not fit anywhere else.

Urgency levels:
- none: No urgency at all. Normal chat or purely passive information.
- low: Informative, requires some future action but with absolutely no time pressure.
- medium: Requires attention soon (e.g., "reply when you can", "your package arrives tomorrow").
- high: Immediate action required. Heavy psychological pressure, threats of account suspension, fines, or very short time limits (e.g., "Act within 24h or lose your account").

EXAMPLES:

Input: ["Hey mom, can you pick me up at 5?", "OFERTA: 50% de descuento en tus gafas de sol. Compra ya en opticasol.es/baja", "URGENT: Your bank account is suspended. Verify your identity immediately at http://secure-bank-login.com"]
Output: [
  {{"reasoning": "Personal communication between family members. No malicious intent or marketing.", "theme": "personal", "urgency_level": "none"}},
  {{"reasoning": "Marketing message from a commercial entity offering a discount. Not deceptive.", "theme": "promotion", "urgency_level": "medium"}},
  {{"reasoning": "High-urgency deceptive message attempting to steal credentials via a fake URL under the guise of account suspension.", "theme": "account_security", "urgency_level": "high"}}
]

Respond ONLY with valid JSON. No markdown, no formatting."""

    def _mock_enrich(self, text: str) -> Dict[str, Any]:
        """Fallback mock enrichment for testing without an API key.

        This uses simple keyword heuristics and is not intended for production.
        """
        text_lower = text.lower()
        theme = "unknown"
        urgency = "none"
        
        # Simple heuristics for mock
        if "paquete" in text_lower or "package" in text_lower or "delivery" in text_lower:
            theme = "delivery"
        elif "banco" in text_lower or "bank" in text_lower or "pago" in text_lower or "eur" in text_lower:
            theme = "banking"
            
        if "urgente" in text_lower or "retenido" in text_lower or "bloqueada" in text_lower or "unauthorized" in text_lower:
            urgency = "high"
        elif "actualice" in text_lower or "verify" in text_lower or "update" in text_lower:
            urgency = "medium"
            
        return {
            "reasoning": "Mock heuristic matched based on keywords.",
            "theme": theme,
            "urgency_level": urgency
        }

    def _mock_enrich_batch(self, texts: list[str]) -> list[Dict[str, Any]]:
        """Fallback mock enrichment for testing without an API key."""
        return [self._mock_enrich(text) for text in texts]

    def annotate_batch(self, texts: list[str]) -> list[Dict[str, Any]]:
        """
        Annotate a batch of texts and return a list of dictionaries.
        """
        if not texts:
            return []

        # Default empty structure in case of failure
        def build_default():
            return {
                "llm_annotated": 0,
                "llm_model": None,
                "prompt_version": None,
                "llm_annotation_date": None,
                "theme": "unknown",
                "urgency_level": "none",
                "reasoning": ""
            }
            
        results = [build_default() for _ in texts]
        
        # If all texts are empty, return early
        if all(not t or len(t.strip()) == 0 for t in texts):
            return results

        llm_response_list = None
        
        if self.use_mock:
            llm_response_list = self._mock_enrich_batch(texts)
        else:
            if self.provider == "groq":
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": json.dumps(texts)}
                    ],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"}
                }
            else:
                payload = {
                    "system_instruction": {
                        "parts": [{"text": self._get_system_prompt()}]
                    },
                    "contents": [{
                        "parts": [{"text": json.dumps(texts)}]
                    }],
                    "generationConfig": {
                        "temperature": 0.0,
                        "response_mime_type": "application/json"
                    }
                }

            for index, api_key in self._iter_api_keys():
                try:
                    if self.provider == "groq":
                        url = "https://api.groq.com/openai/v1/chat/completions"
                        headers = {
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        }
                        response = requests.post(url, headers=headers, json=payload, timeout=20)
                    else:
                        url = (
                            f"https://generativelanguage.googleapis.com/v1beta/models/"
                            f"{self.model_name}:generateContent?key={api_key}"
                        )
                        response = requests.post(url, json=payload, timeout=20)

                    if response.status_code == 200:
                        data = response.json()
                        if self.provider == "groq":
                            response_text = data['choices'][0]['message']['content']
                            # Groq json_object might wrap the array in an object
                            try:
                                parsed = json.loads(response_text)
                                if isinstance(parsed, dict) and len(parsed.keys()) == 1:
                                    parsed_list = list(parsed.values())[0]
                                else:
                                    parsed_list = parsed
                            except Exception:
                                parsed_list = json.loads(response_text)
                        else:
                            response_text = data['candidates'][0]['content']['parts'][0]['text']
                            parsed_list = json.loads(response_text)
                            
                        if isinstance(parsed_list, list) and len(parsed_list) == len(texts):
                            llm_response_list = parsed_list
                            self.api_key_index = index
                            break

                    if response.status_code == 429 or response.status_code >= 500:
                        logger.warning(
                            "%s API rate/availability issue (%s). Rotating API key.",
                            self.provider.capitalize(), response.status_code,
                        )
                        self.api_key_index = (index + 1) % len(self.api_keys)
                        continue

                    logger.error("%s API Error %s: %s", self.provider.capitalize(), response.status_code, response.text)
                    break

                except Exception as e:
                    logger.warning("LLM request failed, rotating API key: %s", str(e))
                    if self.api_keys:
                        self.api_key_index = (index + 1) % len(self.api_keys)
                    continue

            if not llm_response_list:
                raise RuntimeError("All API attempts exhausted or rate limited. Halting execution to prevent mock data pollution. Please resume later.")

        if llm_response_list:
            for i, llm_response in enumerate(llm_response_list):
                # Skip updating empty original texts
                if not texts[i] or len(texts[i].strip()) == 0:
                    continue
                results[i].update({
                    "llm_annotated": 1,
                    "llm_model": self.model_name if not self.use_mock else f"mock-{self.model_name}",
                    "prompt_version": self.PROMPT_VERSION,
                    "llm_annotation_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "theme": str(llm_response.get("theme", "unknown")).lower(),
                    "urgency_level": str(llm_response.get("urgency_level", "none")).lower(),
                    "reasoning": str(llm_response.get("reasoning", ""))
                })
            
        return results

    def annotate(self, text: str) -> Dict[str, Any]:
        """
        Annotate a single text (wrapper around annotate_batch).
        """
        return self.annotate_batch([text])[0]

    # Removed classify_label since it's merged into annotate

if __name__ == "__main__":
    # Test script in mock mode
    annotator = LLMMetadataAnnotator(use_mock=True)
    res = annotator.annotate("Correos: Su paquete esta retenido. Pague aduanas en <URL_HTTPS>")
    print(json.dumps(res, indent=2))
