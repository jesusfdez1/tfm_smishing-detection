import os
import json
import logging
import datetime
import re
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class LLMMetadataAnnotator:
    """
    Enriches SMS messages by inferring language, theme, and urgency.

    Uses a local Hugging Face pipeline for text generation.
    """
    
    PROMPT_VERSION = "v2.0-local-7b"
    MODEL_NAME = "Qwen/Qwen3.6-35B-A3B-FP8"
    
    # Taxonomía cerrada de temáticas
    THEMES = [
        "personal",
        "family_emergency",
        "banking",
        "delivery",
        "account_security",
        "tech_support",
        "promotion",
        "survey",
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
        """Initialize the local Hugging Face text-generation pipeline."""
        self.use_mock = use_mock
        self.model_name = model_name or self.MODEL_NAME
        self._llm_pipe = None

        if not self.use_mock:
            try:
                from transformers import pipeline
                import torch
                
                logger.info("Loading local LLM pipeline: %s", self.model_name)
                # Use device_map="auto" to spread across available GPUs
                self._llm_pipe = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device_map="auto",
                    torch_dtype="auto"
                )
                
                if self._llm_pipe.tokenizer.pad_token is None:
                    self._llm_pipe.tokenizer.pad_token = self._llm_pipe.tokenizer.eos_token
                    
                logger.info("Local LLM pipeline initialized.")
            except Exception as e:
                logger.error("Failed to load local model: %s", e)
                raise RuntimeError(f"Could not initialize local model {self.model_name}")

    def _get_system_prompt(self) -> str:
        """Build the instruction prompt for structured JSON output."""
        return f"""You are an expert cybersecurity analyst annotating an SMS message for a machine learning dataset.
You will receive a single text message. Analyze its semantic meaning and language.

You must return a single JSON object.

For the object, you must include EXACTLY these keys:
- "theme": string, strictly one of the themes listed below.
- "urgency_level": string, strictly one of the urgency levels listed below.
- "language": string, the 2-letter ISO 639-1 code of the language the text is written in (e.g., "en", "es", "fr", "pt"). If the text contains regional slang (like Singlish), use the code of its base language (e.g., "en").

Themes mapping (Strictly MECE):
- personal: Casual conversation, greetings, family matters, or personal questions.
- family_emergency: Fake relatives asking for money or claiming their phone broke (e.g., "Hi mom, this is my new number").
- banking: Banks, wire transfers, credit/debit cards, crypto, or specific financial alerts.
- delivery: Packages, post office, customs fees, shipping tracking.
- account_security: Account locks, suspicious logins, PIN/OTP codes, or identity verification.
- tech_support: Fake technical support, virus alerts, or device infection warnings.
- promotion: Prizes won, lotteries, discounts, aggressive commercial offers, or gifts.
- survey: Requests to complete surveys, give feedback, or participate in studies.
- government: Traffic fines, taxes, public administration notifications.
- job_offer: Recruitment, job offers, work from home opportunities, easy money scams.
- dating_adult: Dating, sexual content, "hot singles", adult contacts.
- subscription: Premium SMS services, horoscopes, ringtones, paid subscriptions.
- service_alert: Medical appointments, utility bills/outages, mobile carrier service reminders.
- unknown: Completely unreadable, lacks context, or does not fit anywhere else.

Urgency levels:
- none: No urgency at all. Normal chat or purely passive information.
- low: Informative, requires some future action but with absolutely no time pressure.
- medium: Requires attention soon (e.g., "reply when you can", "your package arrives tomorrow").
- high: Immediate action required. Heavy psychological pressure, threats of account suspension, fines, or very short time limits.

Respond ONLY with valid JSON. No markdown wrappers, no formatting, just the raw {{ ... }} JSON object.
"""

    def _mock_enrich(self, text: str) -> Dict[str, Any]:
        """Fallback mock enrichment for testing."""
        text_lower = text.lower()
        theme = "unknown"
        urgency = "none"
        
        if "paquete" in text_lower or "package" in text_lower or "delivery" in text_lower:
            theme = "delivery"
        elif "banco" in text_lower or "bank" in text_lower or "pago" in text_lower or "eur" in text_lower:
            theme = "banking"
            
        if "urgente" in text_lower or "retenido" in text_lower or "bloqueada" in text_lower or "unauthorized" in text_lower:
            urgency = "high"
        elif "actualice" in text_lower or "verify" in text_lower or "update" in text_lower:
            urgency = "medium"
            
        return {
            "theme": theme,
            "urgency_level": urgency,
            "language": "es" if "paquete" in text_lower or "banco" in text_lower else "en"
        }

    def _mock_enrich_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self._mock_enrich(text) for text in texts]

    def annotate_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Annotate a batch of texts using the local LLM pipeline.
        """
        if not texts:
            return []

        def build_default():
            return {
                "llm_annotated": 0,
                "llm_model": None,
                "prompt_version": None,
                "llm_annotation_date": None,
                "theme": "unknown",
                "urgency_level": "none",
                "language": ""
            }
            
        results = [build_default() for _ in texts]
        
        if all(not t or len(t.strip()) == 0 for t in texts):
            return results

        llm_response_list = []
        
        if self.use_mock:
            llm_response_list = self._mock_enrich_batch(texts)
        else:
            system_prompt = self._get_system_prompt()
            
            # Format inputs as chat conversations
            chats = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Input: {text}"}
                ]
                for text in texts
            ]
            
            # Run batch inference
            try:
                # Reset call_count to avoid false positive sequential warning from Hugging Face
                if hasattr(self._llm_pipe, "call_count"):
                    self._llm_pipe.call_count = 0
                    
                outputs = self._llm_pipe(
                    chats, 
                    batch_size=len(texts), # Maximize batch size for the given chunk
                    max_new_tokens=200,
                    do_sample=False,
                    return_full_text=False
                )
                
                for out in outputs:
                    generated = out[0]['generated_text']
                    
                    # If the pipeline returns the full chat history, extract the last assistant message
                    if isinstance(generated, list):
                        assistant_msg = next((msg['content'] for msg in reversed(generated) if msg['role'] == 'assistant'), "")
                        generated_text = str(assistant_msg)
                    else:
                        generated_text = str(generated)
                        
                    logger.info(f"Qwen raw output: {generated_text}")
                    
                    # Robust JSON extraction
                    parsed = None
                    json_match = re.search(r"(\{.*\})", generated_text, re.DOTALL)
                    if json_match:
                        try:
                            parsed = json.loads(json_match.group(1))
                        except Exception as e:
                            logger.error(f"JSON parsing failed on {json_match.group(1)}: {e}")
                            
                    if not parsed:
                        parsed = {"theme": "ERROR_PARSING", "urgency_level": "ERROR_PARSING"}
                    
                    llm_response_list.append(parsed)
                    
            except Exception as e:
                logger.error("Local LLM inference failed: %s", e)
                # Fallback to explicit errors instead of defaults
                for _ in texts:
                    llm_response_list.append({"theme": "ERROR_INFERENCE", "urgency_level": "ERROR_INFERENCE"})

        # Format final results
        if llm_response_list:
            for i, llm_response in enumerate(llm_response_list):
                if not texts[i] or len(texts[i].strip()) == 0:
                    continue
                results[i].update({
                    "llm_annotated": 1,
                    "llm_model": self.model_name if not self.use_mock else f"mock-{self.model_name}",
                    "prompt_version": self.PROMPT_VERSION,
                    "llm_annotation_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "theme": str(llm_response.get("theme", "unknown")).lower(),
                    "urgency_level": str(llm_response.get("urgency_level", "none")).lower(),
                    "language": str(llm_response.get("language", "")).lower()[:2]
                })
            
        return results

    def annotate(self, text: str) -> Dict[str, Any]:
        """
        Annotate a single text.
        """
        return self.annotate_batch([text])[0]

    # Removed classify_label since it's merged into annotate

if __name__ == "__main__":
    # Test script in mock mode
    annotator = LLMMetadataAnnotator(use_mock=True)
    res = annotator.annotate("Post: Your package is held. Pay customs at <URL_HTTPS>")
    print(json.dumps(res, indent=2))
