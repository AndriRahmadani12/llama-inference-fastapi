import os
import torch  # type: ignore
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    logging,
)
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        compute_dtype = getattr(torch, "float16")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=compute_dtype,
                low_cpu_mem_usage=True,
            )
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError("Model loading failed")

    def generate_text(self, prompt: str, system_prompt: str = "You are an AI engineer", max_new_tokens: int = 2000, top_k: int = 50, temperature: float = 0.1, top_p: float = 0.9) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        messages.append({"role": "user", "content": prompt})

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        try:
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=True,
                top_k=top_k,
                temperature=temperature,
                top_p=top_p,
            )
            response = outputs[0][input_ids.shape[-1]:]
            output = self.tokenizer.decode(response, skip_special_tokens=True)
            return output
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise RuntimeError("Text generation failed")
