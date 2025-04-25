import os
import requests
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import onnxruntime as ort
from transformers import GPT2Tokenizer
import numpy as np

MODEL_URL = "https://huggingface.co/MSaadAsad/gpt-2-standard/resolve/main/gpt2.onnx"
MODEL_PATH = "gpt-2-vanilla.onnx"
TOKENIZER_PATH = "gpt-2-vanilla_tokenizer"

API_KEY = os.environ["DL_API_KEY"]

app = FastAPI()

# Mount static directory for frontend assets
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Download model and tokenizer if not present
if not os.path.exists(MODEL_PATH):
    print(f"Downloading model from {MODEL_URL}...")
    r = requests.get(MODEL_URL)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
if not os.path.exists(TOKENIZER_PATH):
    print("Downloading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(TOKENIZER_PATH)

# Load ONNX session and tokenizer once at startup
onnx_session = ort.InferenceSession(MODEL_PATH)
tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)

def generate_onnx_gpt2(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    for _ in range(max_length):
        outputs = onnx_session.run(None, {"input_ids": input_ids})
        logits = outputs[0]
        next_token_id = np.argmax(logits[:, -1, :], axis=-1).reshape(-1, 1)
        input_ids = np.concatenate([input_ids, next_token_id], axis=1)
        if next_token_id[0, 0] == tokenizer.eos_token_id:
            break
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

class GenerateRequest(BaseModel):
    prompt: str
    model: str
    max_length: int = 50

class GenerateResponse(BaseModel):
    output: str

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest, http_request: Request):
    api_key = http_request.headers.get("x-api-key")
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    if request.model != "gpt-2-vanilla":
        raise HTTPException(status_code=400, detail="Only 'gpt-2-vanilla' is supported in this configuration.")
    output_text = generate_onnx_gpt2(request.prompt, max_length=request.max_length)
    return GenerateResponse(output=output_text)

@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse("api/static/index.html")
