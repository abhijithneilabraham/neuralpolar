# Neuralpolar

An interactive visual explainer for activation steering in language models.

Neuralpolar has two layers:

1. A browser-only visualizer in `dist/index.html`. It teaches the idea with a
   model map, layer/neuron/dose controls, prompt presets, and pull/baseline/push
   comparisons.
2. An optional local Qwen backend in `backend/steering_server.py`. This loads
   `Qwen/Qwen3-0.6B` on Apple Silicon through PyTorch/MPS and applies a real
   activation intervention during generation.

The visualizer is inspired by the steering-first workflow in
[lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer):
scan units, steer one unit, and compare the resulting rollouts.

## The idea

Inside a transformer layer, an MLP turns a hidden state into an output. The
MLP's `down_proj` has one output direction per intermediate neuron. During a
generation, this project temporarily adds a scaled version of one direction:

```text
steered MLP output = normal MLP output + dose × selected down_proj column
```

Positive `dose` amplifies the selected unit; negative `dose` suppresses it. The
model weights are never edited. The hook exists only while one response is
being generated, then it is removed.

The backend normalizes the intervention using the same basic idea as the
research recipe:

```text
dose × residual_norm / selected_column_norm
```

This makes the dose more comparable across neurons with different weight
magnitudes. A large dose can make the output unstable or repetitive; that is a
useful part of the experiment, not evidence that a neuron literally represents
a complete personality.

## Run the visualizer only

```bash
python3 -m http.server 4173 \
  --directory /Users/abhijithneilabraham/Documents/GitHub/neuralpolar/dist
```

Open <http://localhost:4173>.

The browser-only version is safe to explore without downloading a model. Its
default copy is illustrative. The cards become model-backed only after the
local backend is running and you click **run this dose on local Qwen**.

## Run real local Qwen steering

This path is intended for Apple Silicon. Qwen3-0.6B is used because it fits a
MacBook Air with 8 GB unified memory more comfortably than a 4B model.

In a second terminal, from the repository root:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch 'transformers<5' accelerate
.venv/bin/python backend/steering_server.py
```

The first launch downloads the model from Hugging Face. It starts an API at
`http://127.0.0.1:8787/generate`.

Then keep the static server running and open the visualizer. Select a prompt,
layer, neuron, and dose, then click the Qwen button. The returned generation is
placed into the push card, and the UI reports the model and device used.

You can also call the API directly:

```bash
curl http://127.0.0.1:8787/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What kind of assistant are you?","layer":14,"neuron":1234,"dose":0.2}'
```

## Why the neuron numbers differ

The public research examples use Qwen3.5-4B, whose MLP width is 9,216. This
demo backend uses Qwen3-0.6B, whose MLP is smaller. If a requested neuron is
outside the selected model's range, the backend wraps it into a valid index and
reports both the requested and effective neuron in its JSON response.

## Project layout

```text
dist/index.html              interactive visualizer
backend/steering_server.py   local Qwen inference + steering API
backend/README.md            backend quick reference
```

This is an educational instrument, not a claim that one neuron has one clean
semantic meaning. Reliable interpretation needs repeated prompts, controls,
multiple doses, and careful comparison of outputs.
