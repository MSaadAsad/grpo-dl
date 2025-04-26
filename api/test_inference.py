import os
import numpy as np
import onnxruntime as ort
from transformers import GPT2Tokenizer
import requests

MODELS = {
    "gpt-2-vanilla": {
        "onnx_path": "gpt-2-vanilla.onnx",
        "tokenizer_path": "gpt-2-vanilla_tokenizer",
        "url": "https://huggingface.co/MSaadAsad/dl-grpo/resolve/main/gpt2.onnx"
    },
    "gpt-grpo": {
        "onnx_path": "gpt-grpo.onnx",
        "tokenizer_path": "gpt-grpo_tokenizer",
        "url": "https://huggingface.co/MSaadAsad/dl-grpo/resolve/main/gpt2_grpo.onnx"
    }
}

def ensure_model_and_tokenizer(model_name):
    info = MODELS[model_name]
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

def run_onnx_inference(model_name, prompt, max_length=50):
    info = MODELS[model_name]
    session = ort.InferenceSession(info["onnx_path"])
    tokenizer = GPT2Tokenizer.from_pretrained(info["tokenizer_path"])
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
    prompt = "The quick brown fox"
    for model_name in MODELS:
        ensure_model_and_tokenizer(model_name)
        print(f"\nModel: {model_name}")
        print(f"Prompt: {prompt}")
        generated = run_onnx_inference(model_name, prompt, max_length=50)
        print(f"Generated: {generated}") 