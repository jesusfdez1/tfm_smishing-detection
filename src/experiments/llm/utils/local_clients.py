"""
Local client for interacting with LLM models using Hugging Face transformers.
Supports loading models in half-precision (bfloat16) mapped to available GPUs.
"""

import os
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

class LocalLLMClient:
    def __init__(self, model_id: str, device_map: str = "auto"):
        self.model_id = model_id
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=os.environ.get("HF_TOKEN")
        )
        
        # Use bfloat16 if Ampere+ GPU is available, else float16.
        # Fallback to float32 if CPU-only (not expected in this pipeline).
        dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            
        print(f"[{model_id}] Loading model into VRAM with {dtype} precision...", flush=True)
        
        # We load via pipeline to simplify the generation process
        self.pipeline = pipeline(
            "text-generation",
            model=self.model_id,
            tokenizer=self.tokenizer,
            torch_dtype=dtype,
            device_map=device_map,
            token=os.environ.get("HF_TOKEN")
        )

    def generate(self, prompt: str, system_prompt: str) -> str:
        """
        Generates a response using the model's specific chat template.
        """
        # Format the conversation exactly how the model expects it.
        # This prevents performance degradation from raw prompts.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Ask the tokenizer to apply the correct Jinja template for this model.
        # We don't tokenize yet, just get the formatted string to pass to the pipeline.
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate the output.
        # max_new_tokens is set to 150 to ensure the model has enough space
        # to generate the "reasoning" step in the Chain of Thought experiments.
        outputs = self.pipeline(
            prompt_text,
            max_new_tokens=150,
            max_length=None,
            do_sample=False,   # Deterministic output (Temperature = 0)
            return_full_text=False # Do not return the prompt
        )
        
        generated_text = outputs[0]["generated_text"].strip()
        return generated_text

