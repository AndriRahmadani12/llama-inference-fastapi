from fastapi import APIRouter, HTTPException, FastAPI  # type: ignore
from pydantic import BaseModel , Field # type: ignore
from src.llama_service import LlamaService
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Llama-3.1-Nemotron-70B-Instruct-HF")
service = LlamaService()

class Prompt(BaseModel):
    text: str
    system_prompt: str = "You are an AI assistant"
    max_new_tokens: int = Field(default=1024, gt=0, le=1000000)
    top_k: int = Field(default=50, ge=0, le=1000)
    temperature: float = Field(default=0.4, gt=0.0, le=1.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)


@app.post("/generate", summary="Generate text response", description="This endpoint generates a text response based on the given prompt.")
async def generate(prompt: Prompt):
    try:
        logger.info(f"Received prompt: {prompt.text}")
        response = service.generate_response(
            prompt.text,
            prompt.system_prompt,
            prompt.max_new_tokens,
            prompt.top_k,
            prompt.temperature,
            prompt.top_p
        )
        logger.info("Text generated successfully")
        return {"response": response}
    except Exception as e:
        logger.error(f"Failed to generate text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
