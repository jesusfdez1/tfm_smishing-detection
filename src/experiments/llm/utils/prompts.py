"""
Prompt definitions and utility functions for LLM Zero-Shot and Few-Shot classification.
"""

from typing import List, Dict

# The core instruction for the LLM
SYSTEM_PROMPT = """You are an expert cybersecurity analyst specialized in detecting malicious text messages. 
Your task is to classify the provided SMS message into exactly one of three categories: 'ham' (legitimate/benign), 'spam' (unsolicited commercial/promotional), or 'smishing' (malicious phishing attempts designed to steal data or money).

Output ONLY the category name in lowercase (ham, spam, or smishing). Do NOT output any other text, explanation, or punctuation."""

def build_zero_shot_prompt(message: str) -> str:
    """Builds a zero-shot prompt for a given message."""
    return f"Message: {message}\nCategory:"

def build_few_shot_prompt(message: str, examples: List[Dict[str, str]]) -> str:
    """
    Builds a few-shot prompt for a given message using a list of examples.
    examples should be a list of dicts with 'text' and 'label' keys.
    """
    prompt = ""
    for ex in examples:
        prompt += f"Message: {ex['text']}\nCategory: {ex['label']}\n\n"
        
    prompt += f"Message: {message}\nCategory:"
    return prompt
