"""DSG (Davidsonian Scene Graph) ranker module.

Exports the `DSGRanker` for use with the ranker factory, plus optional helpers.

The `DSGRanker` supports two generation modes:
- Structured pipeline (default): uses few-shot structured prompting to derive
  tuples, dependencies, and questions from the input prompt.
- Lightweight mode (optional): if a `question_generator` callable is provided,
  it will be used to generate questions (and optional dependencies) directly.

You can also inject a custom VQA implementation via `vqa_ranker` which must
implement `.batch_vqa(image, questions, binary=True) -> List[str]`.
"""

from .dsg_score import DSGRanker

# Convenience re-exports for advanced usage
from .generator import generate_dsg_structured
from .vqa import OpenAIVisionVQA, answer_dsg_questions

__all__ = [
    "DSGRanker",
    "generate_dsg_structured",
    "OpenAIVisionVQA",
    "answer_dsg_questions",
]
