# Local Qwen steering backend

This optional backend makes the visualizer real on an Apple Silicon Mac. It
uses Qwen3-0.6B, PyTorch/MPS, and an activation hook on an MLP `down_proj`.
The first run downloads the model from Hugging Face (roughly a few GB,
depending on cache and format). The model weights are never modified.

```bash
cd /Users/abhijithneilabraham/Documents/GitHub/neuralpolar
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch 'transformers<5' accelerate
.venv/bin/python backend/steering_server.py
```

The API accepts `{ "prompt": "...", "layer": 14, "neuron": 123, "dose": 0.8 }` at `POST http://127.0.0.1:8787/generate`.
