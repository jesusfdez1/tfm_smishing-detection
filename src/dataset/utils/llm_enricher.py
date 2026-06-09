import os




import sys
# Workaround for Apptainer stripping the PATH variable set in the bash script.
# We must re-inject the current working directory to allow vLLM to find the 'ninja' wrapper.
os.environ["PATH"] = f"{os.getcwd()}:{os.environ.get('PATH', '')}:/root/.local/bin:{os.path.dirname(sys.executable)}"



import json
import logging
import datetime
import re
import traceback
from typing import Dict, Any, List, Optional
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class LLMMetadataAnnotator:
    """
    Enriches SMS messages by inferring language, theme, and urgency.
    Uses vLLM engine with guided JSON decoding for maximum throughput.
    """

    PROMPT_VERSION = "v1.0-local"
    MODEL_NAME = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"

    # Closed taxonomy for message themes
    THEMES = [
        "personal", "family_emergency", "banking", "delivery",
        "account_security", "tech_support", "promotion", "survey",
        "government", "job_offer", "dating_adult", "subscription",
        "service_alert", "unknown"
    ]

    # Closed taxonomy for urgency levels
    URGENCY_LEVELS = ["none", "low", "medium", "high"]

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.MODEL_NAME
        self._supports_thinking = None  # Cache for Qwen3 enable_thinking check

        logger.info("Attempting vLLM init: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        quant_type = self._detect_quantization()

        engine_kwargs = dict(
            model=self.model_name,
            trust_remote_code=True,
            dtype="auto",
            gpu_memory_utilization=0.90,
            max_model_len=1024,
            tensor_parallel_size=1,
            quantization=quant_type,
            disable_log_stats=True,
        )

        logger.info("vLLM kwargs: %s",
                     {k: v for k, v in engine_kwargs.items() if k != 'model'})

        self.llm = LLM(**engine_kwargs)
        self.sampling_params = self._build_vllm_sampling_params()

        logger.info("✓ vLLM engine initialized (quantization=%s)", quant_type)

    def _detect_quantization(self) -> Optional[str]:
        """Detect quantization type from model name."""
        name_upper = self.model_name.upper()
        if "AWQ" in name_upper:
            return "awq"
        elif "GPTQ" in name_upper:
            return "gptq"
        return None

    def _build_vllm_sampling_params(self):
        """Build vLLM sampling params with guided JSON decoding."""
        json_schema = {
            "type": "object",
            "properties": {
                "theme": {"type": "string", "enum": self.THEMES},
                "urgency_level": {"type": "string", "enum": self.URGENCY_LEVELS},
                "language": {
                    "type": "string",
                    "description": "2-letter ISO 639-1 language code (e.g., 'en', 'es')"
                }
            },
            "required": ["theme", "urgency_level", "language"],
            "additionalProperties": False
        }

        return SamplingParams(
            temperature=0.0,
            max_tokens=100
        )

    def _get_system_prompt(self) -> str:
        """Build the instruction prompt for structured JSON output."""
        return f"""You are an expert cybersecurity analyst annotating SMS messages for a machine learning dataset.

Analyze the semantic meaning and return a JSON object with EXACTLY these keys:
- "theme": strictly one of: {', '.join(self.THEMES)}
- "urgency_level": strictly one of: {', '.join(self.URGENCY_LEVELS)}
- "language": 2-letter ISO 639-1 code (e.g., "en", "es", "fr")

Theme definitions:
- personal: Casual conversation, greetings, family matters.
- family_emergency: Fake relatives asking for money ("Hi mom, new number").
- banking: Banks, transfers, cards, crypto, financial alerts.
- delivery: Packages, post office, customs, shipping.
- account_security: Account locks, suspicious logins, OTP codes.
- tech_support: Fake tech support, virus alerts.
- promotion: Prizes, lotteries, discounts, aggressive offers.
- survey: Requests to complete surveys or feedback.
- government: Traffic fines, taxes, public administration.
- job_offer: Recruitment, work from home, easy money scams.
- dating_adult: Dating, sexual content, adult contacts.
- subscription: Premium SMS, horoscopes, ringtones, paid services.
- service_alert: Medical appointments, utility bills, carrier reminders.
- unknown: Unreadable or doesn't fit elsewhere.

Urgency definitions:
- none: No urgency. Normal chat or passive info.
- low: Informative, future action, no time pressure.
- medium: Attention soon ("reply when you can", "arrives tomorrow").
- high: Immediate action. Psychological pressure, threats, short deadlines.

Return ONLY the raw JSON object. No markdown, no explanations."""

    def _build_chat_prompts(self, texts: List[str]) -> List[str]:
        """Build formatted prompt strings from SMS texts using chat template."""
        system_prompt = self._get_system_prompt()
        chats = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Message: {text}"}
            ]
            for text in texts
        ]

        if self._supports_thinking is None:
            try:
                self.tokenizer.apply_chat_template(
                    chats[0], tokenize=False, add_generation_prompt=True,
                    enable_thinking=False
                )
                self._supports_thinking = True
                logger.info("Qwen3 thinking mode detected → disabled for structured output")
            except (TypeError, Exception):
                self._supports_thinking = False

        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self._supports_thinking:
            kwargs["enable_thinking"] = False

        return [
            self.tokenizer.apply_chat_template(chat, **kwargs)
            for chat in chats
        ]

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """Parse JSON from LLM response, handling thinking tags and markdown."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"```(?:json)?\s*", "", text).strip()
        text = text.rstrip("`").strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def annotate_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Annotate a batch of texts using vLLM.
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

        valid_indices = [i for i, t in enumerate(texts) if t and len(t.strip()) > 0]
        if not valid_indices:
            return results

        valid_texts = [texts[i] for i in valid_indices]

        try:
            prompts = self._build_chat_prompts(valid_texts)
            outputs = self.llm.generate(prompts, self.sampling_params)

            llm_response_list = []
            for out in outputs:
                raw = str(out.outputs[0].text).strip()
                logger.debug("vLLM raw: %s", raw)
                parsed = self._parse_json_response(raw)
                llm_response_list.append(parsed or {
                    "theme": "unknown", "urgency_level": "none", "language": "unknown"
                })
        except Exception as e:
            logger.error("vLLM Inference failed: %s", e)
            logger.error("Traceback:\n%s", traceback.format_exc())
            llm_response_list = [
                {"theme": "unknown", "urgency_level": "none", "language": "unknown"}
                for _ in valid_texts
            ]

        annotation_date = datetime.datetime.now().strftime("%Y-%m-%d")
        for idx, llm_response in zip(valid_indices, llm_response_list):
            results[idx].update({
                "llm_annotated": 1,
                "llm_model": self.model_name,
                "prompt_version": self.PROMPT_VERSION,
                "llm_annotation_date": annotation_date,
                "theme": str(llm_response.get("theme", "unknown")).lower().strip(),
                "urgency_level": str(llm_response.get("urgency_level", "none")).lower().strip(),
                "language": str(llm_response.get("language", "")).lower().strip()[:2]
            })

            if results[idx]["theme"] not in self.THEMES:
                results[idx]["theme"] = "unknown"
            if results[idx]["urgency_level"] not in self.URGENCY_LEVELS:
                results[idx]["urgency_level"] = "none"

        return results

    def annotate(self, text: str) -> Dict[str, Any]:
        """Annotate a single text."""
        return self.annotate_batch([text])[0]


if __name__ == "__main__":
    annotator = LLMMetadataAnnotator()
    res = annotator.annotate(
        "URGENT: Your bank account has been locked. Verify now at http://fake-bank.com"
    )
    print(json.dumps(res, indent=2))
