"""Register default ranker builders.

Importing this module registers the built-in rankers into the shared registry.
"""

from __future__ import annotations

from typing import Any

from ..factory import register_ranker
from ..clipscore import CLIPScoreRanker
from ..decomposed_clipscore import DecomposedCLIPScoreRanker
from ..vqa import VQARanker
from ..instruct_blip_vqa import InstructBLIPVQARanker
from ..dsg import DSGRanker
from ..dsg.generator import generate_dsg_structured


def _build_clipscore(**kwargs: Any) -> CLIPScoreRanker:
    return CLIPScoreRanker(**kwargs)


def _build_decomposed_clipscore(**kwargs: Any) -> DecomposedCLIPScoreRanker:
    # Allow passing an optional nested clip_ranker config or instance.
    clip_ranker = kwargs.pop("clip_ranker", None)
    return DecomposedCLIPScoreRanker(clip_ranker=clip_ranker, **kwargs)


def _build_vqa(**kwargs: Any) -> VQARanker:
    return VQARanker(**kwargs)


def _build_dsg(**kwargs: Any) -> DSGRanker:
    vqa_ranker = kwargs.pop("vqa_ranker", None)
    return DSGRanker(vqa_ranker=vqa_ranker, **kwargs)


def _build_vqa_instructblip(**kwargs: Any) -> InstructBLIPVQARanker:
    return InstructBLIPVQARanker(**kwargs)


def _build_dsg_structured(**kwargs: Any) -> DSGRanker:
    # Register names
    vqa_ranker = kwargs.pop("vqa_ranker", None)
    return DSGRanker(vqa_ranker=vqa_ranker, **kwargs)

    # Register names


register_ranker("clipscore", _build_clipscore)
register_ranker("decomposed_clipscore", _build_decomposed_clipscore)
register_ranker("vqa", _build_vqa)
register_ranker("dsg", _build_dsg)
register_ranker("vqa_instructblip", _build_vqa_instructblip)
register_ranker("dsg_structured", _build_dsg_structured)

__all__ = [
    "CLIPScoreRanker",
    "DecomposedCLIPScoreRanker",
    "VQARanker",
    "DSGRanker",
    "InstructBLIPVQARanker",
]
