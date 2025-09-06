from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from .base import Ranker, ScoreOutput, ImageLike, load_image, to_device
from .defaults import DEFAULT_VQA_PIPELINE_MODEL


class VQARanker(Ranker):
    """Answer questions about an image and score binary answers.

    This ranker wraps a Hugging Face `pipeline('vqa')` model. For binary
    questions, the global score is the average probability of the "yes"
    answer. For open-ended questions, the global score is not well-defined and
    remains 0; answers are still returned in `details`.

    Parameters
    - model_name: VQA model id (default: dandelin/vilt-b32-finetuned-vqa).
    - device: Optional device; if None, selected automatically.
    """

    name = "vqa"

    def __init__(
        self, model_name: Optional[str] = None, device: Optional[str] = None
    ) -> None:
        self.model_name = model_name or DEFAULT_VQA_PIPELINE_MODEL
        self.device = to_device(device)
        self._pipeline: Any = None

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        from transformers import pipeline

        # Device mapping: pipeline expects int or str
        device_arg: int | str | None
        if self.device == "cuda":
            device_arg = 0  # first GPU by default
        elif self.device == "mps":
            device_arg = "mps"
        else:
            device_arg = -1
        self._pipeline = pipeline("vqa", model=self.model_name, device=device_arg)
        return self._pipeline

    @staticmethod
    def _score_yes(answer_items: List[Dict[str, Any]]) -> float:
        # Heuristic: if the top answer is exactly "yes", use its score; else 0.
        if not answer_items:
            return 0.0
        top = answer_items[0]
        ans = str(top.get("answer", "")).strip().lower()
        if ans == "yes":
            return float(top.get("score", 1.0))
        return 0.0

    def score(self, prompt: str, image: ImageLike, **kwargs: Any) -> ScoreOutput:
        """Answer questions about the image.

        kwargs:
        - questions: Optional sequence of questions (defaults to [prompt])
        """
        qa = self._ensure_pipeline()
        pil = load_image(image)

        questions: Sequence[str] = kwargs.get("questions") or [prompt]
        results: List[Dict[str, Any]] = []
        yes_scores: List[float] = []
        for q in questions:
            out = qa(image=pil, question=q)
            # HF can return a dict or list depending on model; normalize to list
            if isinstance(out, dict):
                items = [out]
            else:
                items = list(out)
            results.append({"question": q, "answers": items})
            yes_scores.append(self._score_yes(items))

        global_score = (
            float(sum(yes_scores) / max(1, len(yes_scores))) if yes_scores else 0.0
        )
        details = {"qa": results, "model_name": self.model_name, "device": self.device}
        return ScoreOutput(global_score=global_score, details=details)


__all__ = [
    "VQARanker",
]
