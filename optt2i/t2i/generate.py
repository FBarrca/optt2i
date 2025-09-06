"""t2i generation CLI.

Builds text-to-image models using the factory and optionally generates images.
Intended for direct use via `uv run optt2i.t2i` or `uv run -m optt2i.t2i`.
"""

from __future__ import annotations

import argparse
import inspect
import os
from typing import Any, Dict, List

from . import create_model, list_registered, list_cached


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--builder",
        default="qwen_image_lightning",
        help="Registered builder name to use",
    )
    p.add_argument(
        "--key", default=None, help="Optional cache key (defaults to builder name)"
    )
    p.add_argument(
        "--refresh", action="store_true", help="Force re-create the model instance"
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List registered and cached models, then exit",
    )
    p.add_argument(
        "--prompt",
        default=None,
        help="Prompt to generate an image. If not set, only builds the model",
    )
    p.add_argument("--negative-prompt", default=None, help="Negative prompt (optional)")
    p.add_argument("--width", type=int, default=1024, help="Generation width")
    p.add_argument("--height", type=int, default=1024, help="Generation height")
    p.add_argument(
        "--num-images-per-prompt",
        type=int,
        default=1,
        help="Number of images to generate",
    )
    p.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Optional per-call inference steps (used by many diffusers pipelines)",
    )
    p.add_argument(
        "--out",
        default="output.png",
        help="Output image path (or prefix when multiple images)",
    )


def _add_qwen_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--steps", type=int, default=4, help="Qwen-Image Lightning steps (4 or 8)"
    )
    p.add_argument(
        "--rank", type=int, default=32, help="Nunchaku rank (e.g., 32 or 128)"
    )
    p.add_argument("--true-cfg-scale", type=float, default=1.0, help="True CFG scale")
    p.add_argument(
        "--qwen-torch-dtype",
        default="bfloat16",
        help="Torch dtype for Qwen-Image (e.g., bfloat16, float16, float32)",
    )


def _add_diffusers_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model-id",
        default="runwayml/stable-diffusion-v1-5",
        help="HF model repo or local path for diffusers pipeline",
    )
    p.add_argument(
        "--pipeline-cls",
        default="diffusers.StableDiffusionPipeline",
        help="Dotted class path of the diffusers Pipeline",
    )
    p.add_argument(
        "--torch-dtype",
        default=None,
        help="Torch dtype for diffusers (e.g., float16, bfloat16). If omitted, library default is used.",
    )
    p.add_argument(
        "--device", default=None, help="Device to move pipeline to (e.g., cuda, cpu)"
    )
    p.add_argument(
        "--enable-offload",
        default="auto",
        choices=["auto", "true", "false"],
        help="Enable CPU offload: auto|true|false",
    )


def _builder_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    if args.builder == "qwen_image_lightning":
        return {
            "num_inference_steps": int(args.steps),
            "rank": int(args.rank),
            "true_cfg_scale": float(args.true_cfg_scale),
            "torch_dtype": args.qwen_torch_dtype,
        }
    elif args.builder == "diffusers_pipeline":
        enable_offload: Any
        if args.enable_offload == "true":
            enable_offload = True
        elif args.enable_offload == "false":
            enable_offload = False
        else:
            enable_offload = "auto"
        return {
            "model_id": args.model_id,
            "pipeline_cls": args.pipeline_cls,
            "torch_dtype": args.torch_dtype,
            "device": args.device,
            "enable_offload": enable_offload,
        }
    else:
        return {}


def _call_generate(model: Any, prompt: str, call_kwargs: Dict[str, Any]):
    # Prefer a model-provided generate method; filter kwargs to supported ones
    if hasattr(model, "generate"):
        try:
            sig = inspect.signature(model.generate)
            allowed = {k: v for k, v in call_kwargs.items() if k in sig.parameters}
            return model.generate(prompt, **allowed)
        except Exception:
            pass

    # Fallback: assume diffusers-like call returning object with .images
    call = dict(call_kwargs)
    call["prompt"] = prompt
    out = model(**call)
    return getattr(out, "images", out)


def _save_images(images: List[Any], out_path: str) -> List[str]:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    saved: List[str] = []
    if len(images) == 1:
        images[0].save(out_path)
        saved.append(out_path)
    else:
        root, ext = os.path.splitext(out_path)
        ext = ext or ".png"
        for i, img in enumerate(images):
            p = f"{root}_{i}{ext}"
            img.save(p)
            saved.append(p)
    return saved


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="optt2i.t2i: text-to-image generation CLI"
    )
    _add_common_args(parser)
    _add_qwen_args(parser)
    _add_diffusers_args(parser)

    args = parser.parse_args(argv)

    if args.list:
        print("Registered builders:")
        for n in list_registered():
            print(f"- {n}")
        cached = list_cached()
        if cached:
            print("\nCached models:")
            for n in cached:
                print(f"- {n}")
        return

    kwargs = _builder_kwargs(args)
    model = create_model(args.builder, key=args.key, refresh=args.refresh, **kwargs)

    cache_key = getattr(model, "cache_key", args.key or args.builder)
    print(
        f"Model ready: builder='{args.builder}', cache_key='{cache_key}', type={type(model).__name__}"
    )

    if not args.prompt:
        return

    call_kwargs: Dict[str, Any] = {
        "negative_prompt": args.negative_prompt,
        "width": int(args.width),
        "height": int(args.height),
        "num_images_per_prompt": int(args.num_images_per_prompt),
    }
    if args.num_inference_steps is not None:
        call_kwargs["num_inference_steps"] = int(args.num_inference_steps)

    images = _call_generate(model, args.prompt, call_kwargs)
    if not isinstance(images, list):
        images = [images]

    paths = _save_images(images, args.out)
    for p in paths:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
