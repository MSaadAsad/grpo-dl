import os
import numpy as np
import onnxruntime as ort
from transformers import GPT2Tokenizer

MODEL_PATH = "gpt-2-vanilla.onnx"
TOKENIZER_PATH = "gpt-2-vanilla_tokenizer"
MODEL_URL = "https://huggingface.co/MSaadAsad/gpt-2-standard/resolve/main/gpt2.onnx"

def ensure_model_and_tokenizer():
    if not os.path.exists(MODEL_PATH):
        import requests
        url = MODEL_URL
        print(f"Downloading model from {url}...")
        r = requests.get(url)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
    if not os.path.exists(TOKENIZER_PATH):
        print("Downloading tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.save_pretrained(TOKENIZER_PATH)

def run_onnx_inference(prompt, max_length=50):
    session = ort.InferenceSession(MODEL_PATH)
    tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    for _ in range(max_length):
        outputs = session.run(None, {"input_ids": input_ids})
        logits = outputs[0]
        next_token_id = np.argmax(logits[:, -1, :], axis=-1).reshape(-1, 1)
        input_ids = np.concatenate([input_ids, next_token_id], axis=1)
        if next_token_id[0, 0] == tokenizer.eos_token_id:
            break
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    ensure_model_and_tokenizer()
    prompt = "The quick brown fox"
    print(f"Prompt: {prompt}")
    generated = run_onnx_inference(prompt, max_length=50)
    print(f"Generated: {generated}") 