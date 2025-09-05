"""Default builder registrations for the t2i factory."""

from .factory import register_model
from .builders import build_diffusers_pipeline, build_qwen_image_lightning


def register_defaults() -> None:
    register_model("qwen_image_lightning", build_qwen_image_lightning)
    register_model("diffusers_pipeline", build_diffusers_pipeline)


__all__ = ["register_defaults"]

