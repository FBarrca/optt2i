

Example usage:

```
optt2i --builder qwen_image_lightning --prompt "A beautiful sunset over a calm ocean" --steps 4 --rank 32 --out output.png
```



# optt2i — Text-to-Image Factory

Lightweight factory and cache for text-to-image (t2i) models, with ready‑to‑use builders for both Hugging Face diffusers pipelines and Nunchaku‑backed Qwen‑Image Lightning. Includes a small CLI for quick generation.

## Features

- Registry + in‑memory cache for t2i models
- Built‑in builders:
  - `qwen_image_lightning` (Nunchaku transformer + diffusers `QwenImagePipeline`)
  - `diffusers_pipeline` (generic Hugging Face diffusers pipeline)
- Minimal, composable design — add new builders easily
- CLI entry for quick image generation

## Install

- Python 3.11 or 3.12
- Install dependencies (examples with `uv`):

```bash
uv sync
```

The project pins `torch` via the CUDA nightly index and depends on `diffusers`, `transformers`, and `nunchaku`. Adjust your environment as needed.

## Quickstart — CLI

List registered/cached models:

```bash
uv run optt2i.t2i --list
```

Generate with Qwen‑Image Lightning (Nunchaku):

```bash
uv run optt2i.t2i \
  --builder qwen_image_lightning \
  --steps 4 --rank 32 --qwen-torch-dtype bfloat16 \
  --prompt "A cozy bookstore window" \
  --num-images-per-prompt 1 \
  --out qwen_demo.png
```

Generate with a diffusers pipeline (Stable Diffusion v1.5):

```bash
uv run optt2i.t2i \
  --builder diffusers_pipeline \
  --model-id runwayml/stable-diffusion-v1-5 \
  --pipeline-cls diffusers.StableDiffusionPipeline \
  --torch-dtype float16 --enable-offload auto \
  --prompt "A sunset over mountains" \
  --num-inference-steps 30 \
  --out sd15.png
```

The CLI is also available as a module:

```bash
uv run -m optt2i.t2i --list
```

## Quickstart — Python API

Build and cache a Qwen‑Image pipeline and generate images:

```python
from optt2i.t2i import create_model

pipe = create_model(
    "qwen_image_lightning",
    key="qwen4_r32",
    num_inference_steps=4,
    rank=32,
    true_cfg_scale=1.0,
    torch_dtype="bfloat16",
)

images = pipe.generate("A cozy bookstore window", num_images_per_prompt=2)
images[0].save("qwen_demo.png")
```

Use a generic diffusers pipeline:

```python
from optt2i.t2i import create_model

sd = create_model(
    "diffusers_pipeline",
    key="sd15",
    model_id="runwayml/stable-diffusion-v1-5",
    pipeline_cls="diffusers.StableDiffusionPipeline",
    torch_dtype="float16",
    enable_offload="auto",
)

imgs = sd.generate("A sunset over mountains", num_inference_steps=30)
imgs[0].save("sd15.png")
```

## Factory API

```python
from optt2i.t2i import (
    register_model, create_model, get_cached,
    list_registered, list_cached, clear_cache,
)
```

- `register_model(name, builder)`: Register a builder callable.
- `create_model(name, *, key=None, refresh=False, **kwargs)`: Create or fetch from cache.
- `get_cached(key)`: Retrieve a cached instance.
- `list_registered()`, `list_cached()`, `clear_cache(key=None)`.

## Available Builders

- `qwen_image_lightning(num_inference_steps=4|8, rank=32|128, true_cfg_scale=1.0, torch_dtype="bfloat16")`
  - Nunchaku quantized transformer + diffusers `QwenImagePipeline`
  - Adds `generate(prompt, negative_prompt=None, width=1024, height=1024, num_images_per_prompt=1)`

- `diffusers_pipeline(model_id, pipeline_cls="diffusers.StableDiffusionPipeline", torch_dtype=None, device=None, enable_offload="auto")`
  - Generic HF pipeline with a `generate(...)` helper mirroring common kwargs

## Add A New Builder

1. Create a builder in `optt2i/t2i/builders/your_model.py` that returns an instantiated pipeline/model.
2. Export it from `optt2i/t2i/builders/__init__.py` (optional but convenient).
3. Register it:
   - Temporarily at runtime via `register_model("your_model", build_your_model)`, or
   - Permanently by adding it to `optt2i/t2i/defaults.py` inside `register_defaults()`.

Example skeleton:

```python
# optt2i/t2i/builders/your_model.py
def build_your_model(*, some_arg: int = 1):
    model = ...  # construct your pipeline/model
    # optionally attach a convenience method
    def generate(prompt: str, **kwargs):
        out = model(prompt=prompt, **kwargs)
        return getattr(out, "images", out)
    setattr(model, "generate", generate)
    return model
```

## Notes & Tips

- Offloading: builders attempt CPU offload on low‑VRAM setups. You can override using the CLI `--enable-offload` or builder kwargs.
- VRAM: Qwen‑Image Lightning can run on 3–4 GB VRAM with sequential CPU offload; more VRAM improves speed.
- Heavy dependencies (torch/diffusers/nunchaku) are imported lazily when a builder runs.

## License

MIT

