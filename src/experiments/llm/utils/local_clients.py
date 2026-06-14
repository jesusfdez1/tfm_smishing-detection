"""
Local client for interacting with LLM models using vLLM for high-throughput batched inference.
Supports loading models in half-precision (bfloat16) mapped to available GPUs.
"""

import os
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

class LocalLLMClient:
    def __init__(self, model_id: str, device_map: str = "auto"):
        self.model_id = model_id
        
        # Load the tokenizer to apply the chat template accurately
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=os.environ.get("HF_TOKEN")
        )
        
        # Use bfloat16 if Ampere+ GPU is available, else float16.
        dtype = "float16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = "bfloat16"
            
        print(f"[{model_id}] Loading model into VRAM with vLLM ({dtype})...", flush=True)
        
        # Initialize the vLLM engine
        self.llm = LLM(
            model=self.model_id,
            dtype=dtype,
            trust_remote_code=True,
            # Limit max length to save memory, SMS are short anyway
            max_model_len=2048,
            gpu_memory_utilization=0.90
        )
        
        # Sampling parameters for deterministic generation
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=150
        )

    def generate_batch(self, prompts: list[str], system_prompt: str) -> list[str]:
        """
        Generates responses for a batch of prompts efficiently using vLLM's continuous batching.
        """
        formatted_prompts = []
        for prompt in prompts:
            # Format the conversation exactly how the model expects it.
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts.append(prompt_text)
        
        # Generate outputs in parallel
        # vLLM handles batching and scheduling under the hood automatically
        outputs = self.llm.generate(formatted_prompts, self.sampling_params, use_tqdm=True)
        
        # Extract generated text from each output object
        generated_texts = []
        for output in outputs:
            generated_texts.append(output.outputs[0].text.strip())
            
        return generated_texts
