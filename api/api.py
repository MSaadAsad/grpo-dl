import os
import requests
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import onnxruntime as ort
from transformers import GPT2Tokenizer
import numpy as np

MODEL_INFOS = {
    "gpt-2-vanilla": {
        "url": "https://huggingface.co/MSaadAsad/gpt-2-standard/resolve/main/gpt2.onnx",
        "onnx_path": "gpt-2-vanilla.onnx",
        "tokenizer_path": "gpt-2-vanilla_tokenizer"
    },
    "gpt-grpo": {
        "url": "https://huggingface.co/MSaadAsad/dl-grpo/resolve/main/gpt2_grpo.onnx",
        "onnx_path": "gpt-grpo.onnx",
        "tokenizer_path": "gpt-grpo_tokenizer"
    }
}

app = FastAPI()

# Mount static directory for frontend assets
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Download models and tokenizers if not present, and load sessions/tokenizers
onnx_sessions = {}
tokenizers = {}
for model_name, info in MODEL_INFOS.items():
    if not os.path.exists(info["onnx_path"]):
        print(f"Downloading {model_name} model from {info['url']} ...")
        r = requests.get(info["url"])
        r.raise_for_status()
        with open(info["onnx_path"], "wb") as f:
            f.write(r.content)
    if not os.path.exists(info["tokenizer_path"]):
        print(f"Downloading {model_name} tokenizer ...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.save_pretrained(info["tokenizer_path"])
    onnx_sessions[model_name] = ort.InferenceSession(info["onnx_path"])
    tokenizers[model_name] = GPT2Tokenizer.from_pretrained(info["tokenizer_path"])

def generate_onnx_gpt2(model_name, prompt, max_length=50):
    session = onnx_sessions[model_name]
    tokenizer = tokenizers[model_name]
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    for _ in range(max_length):
        outputs = session.run(None, {"input_ids": input_ids})
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
    if request.model not in MODEL_INFOS:
        raise HTTPException(status_code=400, detail="Invalid model name.")
    output_text = generate_onnx_gpt2(request.model, request.prompt, max_length=request.max_length)
    return GenerateResponse(output=output_text)

@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse("api/static/index.html")
