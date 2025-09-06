from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import math

from .base import Ranker, ScoreOutput, ImageLike, load_image, to_device
from .defaults import DEFAULT_CLIP_MODEL


@dataclass
class _ClipState:
    model_name: str
    device: str
    model: Any
    processor: Any


class CLIPScoreRanker(Ranker):
    """Compute a single CLIPScore between a prompt and an image.

    The score is the cosine similarity between CLIP text and image embeddings,
    linearly mapped to [0, 1]. Values closer to 1 indicate higher consistency.

    Parameters
    - model_name: Hugging Face model id for CLIP (default: openai/clip-vit-base-patch32)
    - device: Optional device string (e.g., "cuda", "cpu"). Auto-detected if None.
    """

    name = "clipscore"

    def __init__(
        self, model_name: str | None = None, device: Optional[str] = None
    ) -> None:
        self._state: Optional[_ClipState] = None
        self.model_name = model_name or DEFAULT_CLIP_MODEL
        self.device = to_device(device)

    def _ensure_model(self) -> _ClipState:
        if self._state is not None:
            return self._state

        from transformers import CLIPModel, CLIPProcessor, SiglipModel, SiglipProcessor

        # Check if this is a SigLIP model
        if "siglip" in self.model_name.lower():
            model = SiglipModel.from_pretrained(self.model_name)
            processor = SiglipProcessor.from_pretrained(self.model_name)
        else:
            model = CLIPModel.from_pretrained(self.model_name)
            processor = CLIPProcessor.from_pretrained(self.model_name)

        model.eval().to(self.device)
        self._state = _ClipState(self.model_name, self.device, model, processor)
        return self._state

    @staticmethod
    def _cosine_to_unit_interval(x: float) -> float:
        # Cosine similarity is typically in [-1, 1]; map to [0, 1]
        return 0.5 * (x + 1.0)

    def score(self, prompt: str, image: ImageLike, **kwargs: Any) -> ScoreOutput:
        state = self._ensure_model()
        pil = load_image(image)

        inputs = state.processor(
            text=[prompt], images=[pil], return_tensors="pt", padding=True
        )
        for k, v in inputs.items():
            inputs[k] = v.to(state.device)

        import torch

        with torch.no_grad():
            outputs = state.model(**inputs)

            # Handle both CLIP and SigLIP model outputs
            if hasattr(outputs, "image_embeds") and hasattr(outputs, "text_embeds"):
                # CLIP model outputs
                image_embeds = outputs.image_embeds  # (B, D)
                text_embeds = outputs.text_embeds  # (B, D)
            else:
                # SigLIP model outputs (different attribute names)
                image_embeds = outputs.image_embeds  # (B, D)
                text_embeds = outputs.text_embeds  # (B, D)

            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            # Batch size 1; take scalar cosine similarity
            cosine_sim = (image_embeds * text_embeds).sum(dim=-1).squeeze().item()

        score = self._cosine_to_unit_interval(float(cosine_sim))
        score = max(0.0, min(1.0, score))
        details: Dict[str, Any] = {
            "cosine_similarity": float(cosine_sim),
            "model_name": state.model_name,
            "device": state.device,
        }
        return ScoreOutput(global_score=score, details=details)


__all__ = [
    "CLIPScoreRanker",
]
