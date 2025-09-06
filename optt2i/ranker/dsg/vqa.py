import base64
import io
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from PIL import Image

from optt2i.ranker.dsg.openai_utils import client, _DEFAULT_MODEL
from optt2i.ranker.dsg.types import QuestionOutput

__all__ = [
    "OpenAIVisionVQA",
    "answer_dsg_questions",
]

# ---------------------------
# Image helpers
# ---------------------------

ImageLike = Union[str, Path, Image.Image, bytes, io.BytesIO]


def _ensure_pil(image: ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image)).convert("RGB")
    if isinstance(image, io.BytesIO):
        return Image.open(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def _pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "WEBP": "image/webp",
    }.get(fmt.upper(), "image/png")
    return f"data:{mime};base64,{b64}"


# ---------------------------
# OpenAI-compatible Vision VQA
# ---------------------------


class OpenAIVisionVQA:
    """
    Vision VQA using an OpenAI-compatible Chat Completions API.

    - Expects a vision-capable `model` (set via env `VQA_OPENAI_MODEL` or fallback to `OPENAI_MODEL`).
    - Accepts images as path, PIL.Image, bytes, or BytesIO.
    - Supports binary (yes/no) or free-form answering.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        system_prompt: str = "You are a precise visual question answering assistant.",
        temperature: float = 0.0,
        max_tokens: int = 50,
    ) -> None:
        env_model = os.getenv("VQA_OPENAI_MODEL") or _DEFAULT_MODEL
        self.model = model or env_model
        if not self.model:
            raise ValueError(
                "No model set. Provide `model=...` or set VQA_OPENAI_MODEL/OPENAI_MODEL."
            )
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_messages(
        self, image: ImageLike, question: str, *, binary: bool
    ) -> List[Dict[str, Any]]:
        img = _ensure_pil(image)
        data_url = _pil_to_data_url(img, fmt="PNG")

        if binary:
            user_text = f"Answer ONLY 'yes' or 'no'.\nQuestion: {question}"
        else:
            user_text = f"Question: {question}\nAnswer concisely."

        # Use OpenAI Chat Completions with image_url data URL for broad compatibility
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]

    def vqa(
        self,
        image: ImageLike,
        question: str,
        *,
        binary: bool = True,
    ) -> str:
        """Answer a single question about an image.

        - If `binary=True`, returns strictly 'yes' or 'no' (lowercase).
        - Otherwise returns a concise free-form string.
        """
        messages = self._build_messages(image, question, binary=binary)
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        out = (resp.choices[0].message.content or "").strip()

        if binary:
            normalized = out.lower().strip().split()[0].strip(".,!")
            if normalized not in {"yes", "no"}:
                # Heuristic fallback: map common variants
                if any(tok in normalized for tok in ["true", "correct"]):
                    return "yes"
                if any(tok in normalized for tok in ["false", "incorrect"]):
                    return "no"
                # Default to conservative 'no' if uncertain
                return "no"
            return normalized

        return out

    def batch_vqa(
        self,
        image: ImageLike,
        questions: Iterable[str],
        *,
        binary: bool = True,
    ) -> List[str]:
        """Answer multiple questions about the same image (sequentially)."""
        return [self.vqa(image, q, binary=binary) for q in questions]


# ---------------------------
# DSG integration helpers
# ---------------------------


def answer_dsg_questions(
    image: ImageLike,
    question_output: Union[QuestionOutput, Mapping[str, Any]],
    *,
    vqa: Optional[OpenAIVisionVQA] = None,
    binary: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run VQA on questions produced by `dsg.query_utils.generate_dsg_structured`.

    Returns a list of records: {id, question, answer}
    """
    if isinstance(question_output, QuestionOutput):
        questions = question_output.model_dump().get("questions", [])
    else:
        questions = dict(question_output).get("questions", [])

    vqa = vqa or OpenAIVisionVQA()
    results: List[Dict[str, Any]] = []
    for item in questions:
        qid = item.get("id")
        qtext = item.get("question")
        ans = vqa.vqa(image, qtext, binary=binary)
        results.append({"id": qid, "question": qtext, "answer": ans})

    return results


# ---------------------------
# CLI for quick testing
# ---------------------------


def _load_questions_from_json(path: Union[str, Path]) -> Mapping[str, Any]:
    import json

    with open(path, "r") as f:
        data = json.load(f)
    # Accept either direct {"questions": [...]} or full QuestionOutput dump
    if isinstance(data, dict) and "questions" in data:
        return data
    raise ValueError("questions JSON must contain a 'questions' key")
