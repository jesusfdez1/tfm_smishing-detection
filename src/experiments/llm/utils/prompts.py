"""
Prompt definitions and utility functions for LLM zero-shot, few-shot, and CoT classification.
"""
from typing import List, Dict

# Experimento 6: Ingeniería de Roles (Persona)
SYSTEM_PROMPT_BASIC = """You are an automated text classifier.
Your task is to classify the provided SMS message into exactly one of three categories: 'ham', 'spam', or 'smishing'."""

SYSTEM_PROMPT_EXPERT = """You are an expert cybersecurity analyst at a telecommunications Security Operations Center.
Your task is to analyze SMS messages and classify them as 'ham' (legitimate), 'spam' (unsolicited marketing), or 'smishing' (phishing/malicious).
Pay close attention to psychological manipulation, extreme urgency, suspicious URLs, and impersonation of entities (banks, postal services, etc.)."""

# Experimento 7: JSON Formatting constraints
JSON_CONSTRAINT_DIRECT = """
Return your answer strictly in valid JSON format with exactly one key "label" containing the string "ham", "spam", or "smishing".
Do NOT include any markdown blocks, explanations, or other text outside the JSON.
Example: {"label": "smishing"}
"""

JSON_CONSTRAINT_COT = """
Return your answer strictly in valid JSON format. You MUST include two keys:
1. "reasoning": A brief step-by-step deduction explaining why the message belongs to the category.
2. "label": Exactly one of "ham", "spam", or "smishing".
Do NOT include any markdown blocks or other text outside the JSON.
Example: {"reasoning": "The message claims a package is blocked and provides a suspicious bit.ly link.", "label": "smishing"}
"""

def get_system_prompt(persona: str = "expert", reasoning: str = "direct") -> str:
    """Returns the combined system prompt based on persona and reasoning mode."""
    base_prompt = SYSTEM_PROMPT_EXPERT if persona == "expert" else SYSTEM_PROMPT_BASIC
    constraint = JSON_CONSTRAINT_COT if reasoning == "cot" else JSON_CONSTRAINT_DIRECT
    return f"{base_prompt}\n\n{constraint}"

def build_zero_shot_prompt(message: str) -> str:
    """Builds a zero-shot prompt for a given message."""
    return f"Message: {message}\n"

def build_few_shot_prompt(message: str, examples: List[Dict[str, str]]) -> str:
    """
    Builds a few-shot prompt for a given message using a list of examples.
    """
    prompt = "Here are some examples of messages and their correct labels:\n\n"
    for ex in examples:
        prompt += f"Example Message: {ex['text']}\nLabel: {ex['label']}\n\n"
        
    prompt += f"Now classify the following message:\nMessage: {message}\n"
    return prompt
