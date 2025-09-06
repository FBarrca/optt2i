from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from .factory import create_ranker
from .base import ScoreOutput, ImageLike


def score_with_methods(
    prompt: str,
    image: ImageLike,
    methods: Iterable[str],
    *,
    ranker_params: Optional[Mapping[str, Dict[str, Any]]] = None,
) -> Dict[str, ScoreOutput]:
    """Compute scores using multiple registered methods.

    - methods: Iterable of registered ranker names, e.g., ["clipscore", "dsg"].
    - ranker_params: Optional per-method kwargs for builder.
    """
    outputs: Dict[str, ScoreOutput] = {}
    for name in methods:
        params = dict((ranker_params or {}).get(name, {}))
        ranker = create_ranker(name, key=name, **params)
        outputs[name] = ranker.score(prompt, image)
    return outputs


__all__ = [
    "score_with_methods",
]
