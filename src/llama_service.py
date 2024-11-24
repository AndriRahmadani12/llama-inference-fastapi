import os
from src.Llama_model_HF import LlamaModel
import logging


logger = logging.getLogger(__name__)

# MODEL_PATH = os.getenv("MODEL_PATH", "../../local-llama-models/Meta-Llama-3.1-8B-Instruct")
MODEL_PATH = os.getenv("MODEL_PATH", "./Llama-3.1-8B-Instruct")

class LlamaService:
    def __init__(self):
        try:
            self.model = LlamaModel(MODEL_PATH)
            logger.info("LlamaService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LlamaService: {str(e)}")
            raise RuntimeError("LlamaService initialization failed")

    def generate_response(self, prompt: str, system_prompt: str, max_new_tokens: int, top_k: int, temperature: float, top_p: float) -> str:
        try:
            return self.model.generate_text(prompt, system_prompt=system_prompt, max_new_tokens=max_new_tokens, top_k=top_k, temperature=temperature, top_p=top_p)
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise
