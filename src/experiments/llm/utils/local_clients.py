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
        
        print(f"[{model_id}] Loading model into VRAM with vLLM (auto dtype)...", flush=True)
        
        # Initialize the vLLM engine
        # We MUST use dtype="auto" and NOT call torch.cuda.is_available() before this,
        # otherwise CUDA initializes in the parent process, forcing vLLM to use 'spawn'
        # which crashes silently inside Apptainer containers due to /dev/shm limits.
        self.llm = LLM(
            model=self.model_id,
            dtype="auto",
            trust_remote_code=True,
            # Limit max length to save memory, SMS are short anyway
            max_model_len=2048,
            gpu_memory_utilization=0.90,
            enforce_eager=True, # Avoid CUDA graph capture crashes on some clusters
            tensor_parallel_size=torch.cuda.device_count(), # ¡Magia Multi-GPU!
            disable_custom_all_reduce=True # Evita el cuelgue CUSTOM cuando P2P está desactivado en HPC
        )
        
        # Sampling parameters for deterministic generation
        temp = 0.3 if "nemo" in self.model_id.lower() else 0.0
        self.sampling_params = SamplingParams(
            temperature=temp,
            max_tokens=1024 # Increased significantly to allow DeepSeek-R1 to finish its <think> block
        )

    def generate_batch(self, prompts: list[str], system_prompt: str) -> list[str]:
        """
        Generates responses for a batch of prompts efficiently using vLLM's continuous batching.
        """
        formatted_prompts = []
        for prompt in prompts:
            # DeepSeek-R1 and OlmoE prefer NO system prompt (instructions inside user prompt)
            if "deepseek-r1" in self.model_id.lower() or "olmoe" in self.model_id.lower():
                messages = [
                    {"role": "user", "content": f"{system_prompt}\n\n{prompt}"}
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            
            # Apply chat template
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True
            }
            if "qwen3.5" in self.model_id.lower():
                template_kwargs["enable_thinking"] = False
                
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                **template_kwargs
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
