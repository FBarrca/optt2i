from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .base import Ranker, ScoreOutput, ImageLike, load_image, to_device


class InstructBLIPVQARanker(Ranker):
    """VQA using Instruct-BLIP (Salesforce).

    This implementation generates short answers to yes/no questions and scores
    1.0 for "yes" (case-insensitive, prefix match) and 0.0 otherwise.

    Parameters
    - model_name: e.g., "Salesforce/instructblip-flan-t5-xl" or compatible
    - device: Optional device string (auto-detected if None)
    - max_new_tokens: Limit for generation
    - force_yes_no: If True, enforce a strict yes/no answer in the prompt
    """

    name = "vqa_instructblip"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        *,
        max_new_tokens: int = 5,
        force_yes_no: bool = True,
    ) -> None:
        self.model_name = model_name or "Salesforce/instructblip-flan-t5-xl"
        self.device = to_device(device)
        self.max_new_tokens = int(max_new_tokens)
        self.force_yes_no = bool(force_yes_no)

        self._model = None
        self._processor = None

    def _ensure_model(self):
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        from transformers import InstructBlipForConditionalGeneration, AutoProcessor
        import torch

        model = InstructBlipForConditionalGeneration.from_pretrained(self.model_name)
        processor = AutoProcessor.from_pretrained(self.model_name)
        model.eval().to(self.device)
        self._model = model
        self._processor = processor
        return model, processor

    @staticmethod
    def _is_yes(text: str) -> bool:
        t = text.strip().lower()
        return t.startswith("yes")

    def _answer_one(self, image, question: str) -> Dict[str, Any]:
        model, processor = self._ensure_model()
        import torch

        prompt = question
        if self.force_yes_no:
            # Encourage concise binary answers
            prompt = f"Question: {question}\nAnswer with a single word: yes or no."

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(
            self.device
        )
        output_ids = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        answer = text.strip()
        yes_score = 1.0 if self._is_yes(answer) else 0.0
        return {"answer": answer, "score": yes_score}

    def score(self, prompt: str, image: ImageLike, **kwargs: Any) -> ScoreOutput:
        pil = load_image(image)
        questions: Sequence[str] = kwargs.get("questions") or [prompt]

        results: List[Dict[str, Any]] = []
        yes_scores: List[float] = []
        for q in questions:
            ans = self._answer_one(pil, q)
            results.append({"question": q, "answers": [ans]})
            yes_scores.append(float(ans.get("score", 0.0)))

        global_score = (
            float(sum(yes_scores) / max(1, len(yes_scores))) if yes_scores else 0.0
        )
        details = {
            "qa": results,
            "model_name": self.model_name,
            "device": self.device,
        }
        return ScoreOutput(global_score=global_score, details=details)


__all__ = [
    "InstructBLIPVQARanker",
]
