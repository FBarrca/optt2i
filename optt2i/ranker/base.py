from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional at import time
    Image = Any  # type: ignore

ImageLike = Union["Image.Image", str]


@dataclass
class ScoreOutput:
    """Standardized output of a ranker score call.

    - global_score: Overall scalar in [0, 1] (best effort normalization).
    - details: Method-specific structured details for interpretability.
    - extras: Optional metadata (e.g., per-token logits) not meant for scoring.
    """

    global_score: float
    details: Dict[str, Any]
    extras: Optional[Dict[str, Any]] = None


class Ranker(ABC):
    """Abstract base class for prompt-image rankers."""

    name: str = "base"

    @abstractmethod
    def score(self, prompt: str, image: ImageLike, **kwargs: Any) -> ScoreOutput:
        """Compute a consistency score given a prompt and an image.

        Implementations should return a `ScoreOutput` whose `global_score` is
        higher for better prompt-image consistency.
        """

    def __call__(self, prompt: str, image: ImageLike, **kwargs: Any) -> ScoreOutput:
        return self.score(prompt, image, **kwargs)


def load_image(image: ImageLike) -> "Image.Image":
    """Load a PIL image from a PIL.Image, file path, or URL-like path.

    Note: URL fetching is not implemented to avoid network requirements.
    """
    if Image is Any:
        raise RuntimeError("Pillow is required to load images but is not available")

    if hasattr(image, "__class__") and image.__class__.__name__ == "Image":  # type: ignore[attr-defined]
        return image  # type: ignore[return-value]
    if isinstance(image, str):
        # Avoid importing requests; assume local path.
        from PIL import Image as PILImage

        return PILImage.open(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def to_device(device_preference: Optional[str] = None) -> str:
    """Pick a torch device string; falls back to CPU if unavailable."""
    try:
        import torch
    except Exception:
        return "cpu"

    if device_preference:
        return device_preference
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


__all__ = [
    "ScoreOutput",
    "Ranker",
    "ImageLike",
    "load_image",
    "to_device",
]
