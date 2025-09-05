from __future__ import annotations

from typing import Any, List, Optional


def build_qwen_image_lightning(
    *,
    num_inference_steps: int = 4,
    rank: int = 32,
    true_cfg_scale: float = 1.0,
    torch_dtype: str = "bfloat16",
) -> Any:
    """Build a Qwen-Image Lightning pipeline backed by a Nunchaku quantized transformer.

    Mirrors the setup used in sample code but encapsulated in a builder.
    Returns a configured diffusers QwenImagePipeline instance.
    """
    import math
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline

    from nunchaku.models.transformers.transformer_qwenimage import (
        NunchakuQwenImageTransformer2DModel,
    )
    from nunchaku.utils import get_gpu_memory, get_precision

    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    precision = get_precision()
    model_paths = {
        4: f"nunchaku-tech/nunchaku-qwen-image/svdq-{precision}_r{rank}-qwen-image-lightningv1.0-4steps.safetensors",
        8: f"nunchaku-tech/nunchaku-qwen-image/svdq-{precision}_r{rank}-qwen-image-lightningv1.1-8steps.safetensors",
    }
    if num_inference_steps not in model_paths:
        raise ValueError(
            f"Unsupported num_inference_steps={num_inference_steps}. Supported: {sorted(model_paths.keys())}"
        )

    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
        model_paths[num_inference_steps]
    )
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    resolved_dtype = dtype_map.get(torch_dtype.lower())
    if resolved_dtype is None:
        raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")

    # check if the gpu is RTX 5000 series 
    # https://nunchaku.tech/docs/nunchaku/usage/attention.html
    if torch.cuda.get_device_name(0) == "RTX 5000":
        transformer.set_attention_impl("nunchaku-fp16")  # set attention implementation to fp16

    pipe = QwenImagePipeline.from_pretrained(
        "Qwen/Qwen-Image", transformer=transformer, scheduler=scheduler, torch_dtype=resolved_dtype
    )

    # Offload heuristics matching the example
    if get_gpu_memory() > 18:
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass
    else:
        try:
            transformer.set_offload(True)
            pipe._exclude_from_cpu_offload.append("transformer")
            pipe.enable_sequential_cpu_offload()
        except Exception:
            pass

    # --- Workaround helper for the Nunchuku bug on num_images_per_prompt != 1 ---
    # Always call the pipeline with num_images_per_prompt=1, and if the caller asked
    # for more, loop multiple runs to collect N images.
    _warned_multi = {"done": False}

    def _generate(
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
    ) -> List[Any]:
        """
        Generate images with a defensive workaround:
        - Forces the underlying call to num_images_per_prompt=1 to avoid crashes.
        - If num_images_per_prompt > 1 was requested, runs the pipeline multiple times.
        - Optional `seed` enables deterministic results; per-image seed is (seed + i).
        """
        import torch

        # How many images the caller *wants*
        requested = int(num_images_per_prompt or 1)
        if requested < 1:
            requested = 1

        # Let the user know (once) we're using the multi-run fallback.
        if requested > 1 and not _warned_multi["done"]:
            print(
                "[nunchaku workaround] Multi-image requested; forcing num_images_per_prompt=1 "
                "and looping per image to avoid the known crash."
            )
            _warned_multi["done"] = True

        # Pick the device the pipeline will actually use
        exec_device = getattr(
            pipe, "_execution_device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        images: List[Any] = []
        for i in range(requested):
            gen = None
            if seed is not None:
                gen = torch.Generator(device=exec_device).manual_seed(int(seed) + i)

            # Always pass num_images_per_prompt=1 to avoid the crash
            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                num_images_per_prompt=1,
                generator=gen,
            )
            # `out.images` is a list; we take the first (and only) image per run
            if getattr(out, "images", None):
                images.append(out.images[0])

        return images

    # Expose a consistent method if users want to call via factory
    setattr(pipe, "generate", _generate)
    setattr(pipe, "cache_key", f"qwen_image_lightning_steps{num_inference_steps}_r{rank}_{precision}")
    return pipe


__all__ = ["build_qwen_image_lightning"]


if __name__ == "__main__":
    # Test the Qwen Image Lightning pipeline
    print("Building Qwen Image Lightning pipeline...")

    try:
        # Build the pipeline with default parameters
        pipe = build_qwen_image_lightning(
            num_inference_steps=4,
            rank=32,
            true_cfg_scale=1.0,
            torch_dtype="bfloat16"
        )
        print("✓ Pipeline built successfully!")
        print(f"✓ Cache key: {pipe.cache_key}")

        # Test image generation
        print("\nGenerating test images with multi-run fallback...")
        prompt = "A beautiful landscape with mountains and a lake at sunset"

        # Request 3 images; workaround will loop 3× with num_images_per_prompt=1 internally
        images = pipe.generate(
            prompt=prompt,
            width=1024,
            height=1024,
            num_images_per_prompt=3,
            seed=1234,  # optional; remove for non-deterministic
        )

        print(f"✓ Generated {len(images)} image(s) successfully!")

        # Save the first image if PIL is available
        try:
            if images:
                images[0].save("test_qwen_image_lightning.png")
                print("✓ Test image saved as 'test_qwen_image_lightning.png'")
        except Exception as e:
            print(f"Note: Could not save image: {e}")

    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
