"""
API clients for interacting with LLM models.
Includes OpenAI (GPT), Gemini, and DeepSeek interfaces with retry logic.
"""

import os
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# We use standard exception types to trigger retries.
class APIRateLimitError(Exception): pass
class APIServerError(Exception): pass

# Import required libraries
import openai
from google import genai
from google.genai import types

class BaseLLMClient:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str, system_prompt: str) -> str:
        raise NotImplementedError

class OpenAIClient(BaseLLMClient):
    def __init__(self, model_name: str, api_key: str = None, base_url: str = None):
        super().__init__(model_name)
        # Use provided key, or fallback to environment variable
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
            
        kwargs = {"api_key": self.api_key}
        if base_url:
            kwargs["base_url"] = base_url
            
        self.client = openai.OpenAI(**kwargs)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type((APIRateLimitError, APIServerError))
    )
    def generate(self, prompt: str, system_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0, # Deterministic answers
                max_tokens=10    # We only need one word (ham, spam, smishing)
            )
            return response.choices[0].message.content.strip().lower()
        except openai.RateLimitError as e:
            raise APIRateLimitError(str(e))
        except openai.APIError as e:
            raise APIServerError(str(e))
        except Exception as e:
            raise e

class DeepSeekClient(OpenAIClient):
    """DeepSeek uses an OpenAI compatible API."""
    def __init__(self, model_name: str = "deepseek-chat"):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables.")
            
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

class GeminiClient(BaseLLMClient):
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        super().__init__(model_name)
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            keys = os.environ.get("GEMINI_API_KEYS", "")
            if keys:
                self.api_key = keys.split(",")[0].strip()
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
        self.client = genai.Client(api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type((APIRateLimitError, APIServerError))
    )
    def generate(self, prompt: str, system_prompt: str) -> str:
        try:
            config = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=10,
                system_instruction=system_prompt,
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            return response.text.strip().lower()
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "rate limit" in err_str:
                raise APIRateLimitError(str(e))
            if "500" in err_str or "503" in err_str:
                raise APIServerError(str(e))
            raise e
