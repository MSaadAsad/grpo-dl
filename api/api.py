import os
import requests
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import onnxruntime as ort
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

MODEL_URLS = {
    "gpt-2-vanilla": "<HF_ONNX_LINK_GPT2_VANILLA>",
    "gpt-2-ppo": "<HF_ONNX_LINK_GPT2_PPO>",
    "gpt-grpo": "<HF_ONNX_LINK_GPT_GRPO>"
}
MODEL_PATHS = {
    name: f"{name}.onnx" for name in MODEL_URLS
}

model_sessions = {}

API_KEY = os.environ["DL_API_KEY"]

def download_model(model_name):
    model_url = MODEL_URLS[model_name]
    model_path = MODEL_PATHS[model_name]
    if not os.path.exists(model_path):
        print(f"Downloading {model_name} model...")
        response = requests.get(model_url)
        response.raise_for_status()
        with open(model_path, "wb") as f:
            f.write(response.content)
        print(f"Model {model_name} downloaded.")
    return model_path

def get_onnx_session(model_name):
    if model_name not in model_sessions:
        model_path = download_model(model_name)
        try:
            model_sessions[model_name] = ort.InferenceSession(model_path)
        except Exception as e:
            print(f"Failed to load ONNX model {model_name}: {e}")
            return None
    return model_sessions[model_name]

app = FastAPI()

# Mount static directory for frontend assets
app.mount("/static", StaticFiles(directory="api/static"), name="static")

class GenerateRequest(BaseModel):
    prompt: str
    model: str

class GenerateResponse(BaseModel):
    output: str

# NOTE: Replace this with proper GPT-2 tokenization for real use
# This is a placeholder for demonstration only
def simple_tokenize(text):
    return [ord(c) for c in text]

def simple_detokenize(tokens):
    return ''.join([chr(t) for t in tokens if t < 128])

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest, http_request: Request):
    api_key = http_request.headers.get("x-api-key")
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    if request.model not in MODEL_URLS:
        raise HTTPException(status_code=400, detail="Invalid model name.")
    ort_session = get_onnx_session(request.model)
    if ort_session is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    # Tokenize input (placeholder logic)
    input_ids = simple_tokenize(request.prompt)
    import numpy as np
    input_array = np.array([input_ids], dtype=np.int64)
    # Run ONNX model (input name may need adjustment)
    try:
        outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_array})
        # Assume output is a sequence of token ids
        output_tokens = outputs[0][0].tolist()
        output_text = simple_detokenize(output_tokens)
        return GenerateResponse(output=output_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ONNX inference failed: {e}")

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GPT-2 ONNX Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 2em; }
            #output { margin-top: 1em; padding: 1em; border: 1px solid #ccc; }
        </style>
    </head>
    <body>
        <h1>GPT-2 ONNX Demo</h1>
        <form id=\"gen-form\">
            <label for=\"model\">Model:</label><br>
            <select id=\"model\" name=\"model\">
                <option value=\"gpt-2-vanilla\">gpt-2-vanilla</option>
                <option value=\"gpt-2-ppo\">gpt-2-ppo</option>
                <option value=\"gpt-grpo\">gpt-grpo</option>
            </select><br><br>
            <label for=\"prompt\">Prompt:</label><br>
            <input type=\"text\" id=\"prompt\" name=\"prompt\" style=\"width: 300px;\" required><br><br>
            <button type=\"submit\">Generate</button>
        </form>
        <div id=\"output\"></div>
        <script>
        document.getElementById('gen-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            const prompt = document.getElementById('prompt').value;
            const model = document.getElementById('model').value;
            document.getElementById('output').innerText = 'Generating...';
            const response = await fetch('/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt, model })
            });
            if (response.ok) {
                const data = await response.json();
                document.getElementById('output').innerText = data.output;
            } else {
                document.getElementById('output').innerText = 'Error: ' + response.statusText;
            }
        });
        </script>
    </body>
    </html>
    """
