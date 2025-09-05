from __future__ import annotations

from typing import Any, Dict, Optional


def build_diffusers_pipeline(
    *,
    model_id: str,
    pipeline_cls: str = "diffusers.StableDiffusionPipeline",
    torch_dtype: Optional[str] = None,
    enable_offload: bool | str = "auto",
    device: Optional[str] = None,
    from_pretrained_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Build a generic Hugging Face diffusers pipeline.

    - model_id: HF model repository ID or local path.
    - pipeline_cls: Dotted import path to a diffusers Pipeline class (e.g., 'diffusers.StableDiffusionPipeline').
    - torch_dtype: Optional dtype string (e.g., 'float16', 'bfloat16'). If None, uses library default.
    - enable_offload: True/False or 'auto' to enable offloading heuristics.
    - device: Optional device string (e.g., 'cuda', 'cpu', 'cuda:0'). If None, diffusers decides.
    - from_pretrained_kwargs: Extra kwargs forwarded to from_pretrained.
    """
    import importlib

    fp_kwargs = dict(from_pretrained_kwargs or {})

    # Resolve dtype lazily to avoid importing torch unless needed
    if torch_dtype is not None:
        import torch as _torch
        dtype_map = {
            "float16": _torch.float16,
            "fp16": _torch.float16,
            "bfloat16": _torch.bfloat16,
            "bf16": _torch.bfloat16,
            "float32": _torch.float32,
            "fp32": _torch.float32,
        }
        resolved_dtype = dtype_map.get(str(torch_dtype).lower())
        if resolved_dtype is None:
            raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")
        fp_kwargs["torch_dtype"] = resolved_dtype

    if "." not in pipeline_cls:
        pipeline_cls = f"diffusers.{pipeline_cls}"

    module_name, class_name = pipeline_cls.rsplit(".", 1)
    module = importlib.import_module(module_name)
    PipelineClass = getattr(module, class_name)

    pipe = PipelineClass.from_pretrained(model_id, **fp_kwargs)

    if enable_offload:
        # Respect simple heuristic if 'auto': try to offload on limited VRAM setups.
        try:
            if enable_offload == "auto":
                try:
                    import torch as _torch

                    vram_gb = 0
                    if _torch.cuda.is_available():
                        idx = _torch.cuda.current_device()
                        vram_gb = _torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
                    if vram_gb and vram_gb < 20:
                        pipe.enable_model_cpu_offload()
                except Exception:
                    # Best-effort heuristic; ignore failures
                    pass
            else:
                pipe.enable_model_cpu_offload()
        except Exception:
            # Some pipelines may not support offload; ignore
            pass

    if device:
        try:
            pipe.to(device)
        except Exception:
            # Some pipelines route device per-call; ignore if .to() unsupported
            pass

    # Provide a thin convenience wrapper to return images directly
    def _generate(
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        num_images_per_prompt: int = 1,
        **call_kwargs: Any,
    ):
        call = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
        )
        if width is not None:
            call["width"] = width
        if height is not None:
            call["height"] = height
        if num_inference_steps is not None:
            call["num_inference_steps"] = num_inference_steps
        call.update(call_kwargs)
        out = pipe(**call)
        return getattr(out, "images", out)

    try:
        setattr(pipe, "generate", _generate)
        setattr(pipe, "cache_key", f"{pipeline_cls}:{model_id}")
    except Exception:
        pass

    return pipe


__all__ = ["build_diffusers_pipeline"]
