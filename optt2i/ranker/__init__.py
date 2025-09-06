"""Ranker factory and scoring interfaces.

This package provides a simple factory/registry for prompt-image consistency
scorers ("rankers") and several reference implementations:

- CLIPScore: single scalar similarity between prompt and image.
- Decomposed CLIPScore: per-noun-phrase CLIPScore with global average.
- VQA: answer binary or open-ended questions over an image.
- DSG (Davidsonian Scene Graph): generate simple binary questions from the
  prompt and answer them with a VQA model to compute a global score.

Models are lazily loaded on first use. Heavy model downloads require an
environment with access to the corresponding weights; this package only
provides the wiring and can run fully offline if models are pre-cached.
"""

from .factory import (
    register_ranker,
    is_registered,
    list_registered,
    list_cached,
    get_cached,
    clear_cache,
    create_ranker,
)
from .utils import score_with_methods
from .llm import (
    get_openai_client,
    llm_noun_phrase_extractor,
    llm_dsg_question_generator,
)

# Auto-register built-in rankers on import
from . import builders as _builtin_builders  # noqa: F401
from .dsg.generator import generate_dsg_structured

__all__ = [
    "register_ranker",
    "is_registered",
    "list_registered",
    "list_cached",
    "get_cached",
    "clear_cache",
    "create_ranker",
    "score_with_methods",
    "get_openai_client",
    "llm_noun_phrase_extractor",
    "llm_dsg_question_generator",
    "generate_dsg_structured",
]
