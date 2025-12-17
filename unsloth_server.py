import os
import sys
import torch
import argparse
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from unsloth import FastLanguageModel, unsloth_logger

# Suppress Unsloth's welcome message on startup
os.environ["UNSLOTH_SUPPRESS_STDOUT"] = "true"

# Redirect logger to stderr
unsloth_logger.remove()
unsloth_logger.add(sys.stderr)

# Global variables to hold the model and tokenizer
model = None
tokenizer = None

# --- FastAPI App Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This block runs on startup
    global model, tokenizer
    print("Starting Unsloth server lifespan...", file=sys.stderr)

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="FastAPI server for Unsloth model inference.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base model.")
    parser.add_argument("--lora_model_path", type=str, default=None, help="Optional path to the LoRA adapter.")
    # Port is handled by uvicorn, not needed here
    args, _ = parser.parse_known_args() # Use parse_known_args to ignore uvicorn args

    print(f"Loading base model: {args.base_model_path}", file=sys.stderr)
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        if args.lora_model_path and os.path.isdir(args.lora_model_path):
            print(f"Loading and applying LoRA adapter from: {args.lora_model_path}", file=sys.stderr)
            model.load_adapter(args.lora_model_path)
            print("LoRA adapter applied successfully.", file=sys.stderr)

        model.eval()
        print("Model loaded successfully.", file=sys.stderr)

    except Exception as e:
        print(f"FATAL: Failed to load model. Error: {e}", file=sys.stderr)
        # In a real app, you might want to exit or handle this more gracefully.
        # For now, the server will start but endpoints will fail.

    yield
    # This block runs on shutdown
    print("Shutting down Unsloth server...", file=sys.stderr)
    model = None
    tokenizer = None


app = FastAPI(lifespan=lifespan)

# --- Pydantic Models for Request/Response ---
class GenerateRequest(BaseModel):
    prompt: str
    history: list = []
    is_cpt_model: bool = False

class GenerateResponse(BaseModel):
    response: str


# --- API Endpoints ---
@app.get("/health")
async def health_check():
    """Health check endpoint to verify the server is running and the model is loaded."""
    if model is not None and tokenizer is not None:
        return {"status": "ok", "message": "Server and model are ready."}
    else:
        raise HTTPException(status_code=503, detail="Server is running, but model is not loaded.")

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generates text based on a prompt and conversation history."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Cannot process request.")

    try:
        if request.is_cpt_model:
            final_prompt = request.prompt
            input_ids = tokenizer(final_prompt, return_tensors="pt").input_ids.to("cuda")
        else:
            conversation = request.history + [{"role": "user", "content": request.prompt}]
            final_prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(final_prompt, return_tensors="pt").input_ids.to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=256,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][input_ids.shape[1]:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return GenerateResponse(response=response_text)

    except Exception as e:
        print(f"Error during generation: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"An error occurred during text generation: {str(e)}")

# This part is for direct execution, but we'll use uvicorn via docker compose exec
if __name__ == "__main__":
    import uvicorn
    # This is for local debugging only. The app will be started by `uvicorn unsloth_server:app ...`
    # Arguments need to be passed via command line, e.g.:
    # uvicorn unsloth_server:app --host 0.0.0.0 --port 8003 -- --base_model_path ./base_models/your_model
    uvicorn.run(app, host="0.0.0.0", port=8003)
