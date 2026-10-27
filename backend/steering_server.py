"""Tiny local Qwen steering server.

This is intentionally a teaching implementation: it hooks the selected MLP
down projection and adds a scaled output-column vector during generation.
Nothing is written back to the model weights.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("QWEN_MODEL", "Qwen/Qwen3-0.6B")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32

print(f"Loading {MODEL_ID} on {DEVICE} …", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
model.to(DEVICE).eval()
print("Model ready.", flush=True)


def generate(prompt: str, layer: int, neuron: int, dose: float):
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt").to(DEVICE)
    target_layer = model.model.layers[layer]
    down_proj = target_layer.mlp.down_proj
    requested_neuron = neuron
    neuron = neuron % down_proj.in_features
    column = down_proj.weight[:, neuron].detach().to(dtype=DTYPE, device=DEVICE)

    def steer_hook(_module, _args, output):
        # Match the research recipe: dose × residual-norm / column-norm.
        residual_norm = output.detach().float().norm(dim=-1).median().clamp_min(1e-5)
        column_norm = column.float().norm().clamp_min(1e-5)
        delta = column * (dose * residual_norm / column_norm)
        if isinstance(output, tuple):
            first, *rest = output
            return (first + delta, *rest)
        return output + delta

    handle = down_proj.register_forward_hook(steer_hook)
    try:
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    finally:
        handle.remove()
    new_tokens = generated[0, inputs.input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()

    def do_POST(self):
        if self.path != "/generate":
            self._send(404, {"error": "not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(size))
            text = generate(str(data.get("prompt", "")), int(data.get("layer", 14)), int(data.get("neuron", 0)), float(data.get("dose", 0)))
            self._send(200, {"text": text, "model": MODEL_ID, "device": DEVICE, "requested_neuron": int(data.get("neuron", 0)), "effective_neuron": int(int(data.get("neuron", 0)) % model.model.layers[int(data.get("layer", 14))].mlp.down_proj.in_features)})
        except Exception as exc:
            self._send(500, {"error": str(exc)})


if __name__ == "__main__":
    print("Steering API: http://127.0.0.1:8787/generate", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8787), Handler).serve_forever()
