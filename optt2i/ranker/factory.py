"""Registry and cache for prompt-image rankers (scorers).

A ranker is any object exposing a `score(prompt, image, **kwargs)` method and
returning a structured dict with at least a `global_score` float.

This mirrors the minimal factory design used by `optt2i.t2i.factory`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

_REGISTRY: Dict[str, Callable[..., Any]] = {}
_CACHE: Dict[str, Any] = {}


def register_ranker(name: str, builder: Callable[..., Any]) -> None:
    """Register a ranker builder by name."""
    if not isinstance(name, str) or not name:
        raise ValueError("Ranker name must be a non-empty string")
    _REGISTRY[name] = builder


def is_registered(name: str) -> bool:
    return name in _REGISTRY


def list_registered() -> List[str]:
    return sorted(_REGISTRY.keys())


def list_cached() -> List[str]:
    return sorted(_CACHE.keys())


def get_cached(key: str) -> Any:
    if key not in _CACHE:
        raise KeyError(f"No cached ranker under key '{key}'")
    return _CACHE[key]


def clear_cache(key: Optional[str] = None) -> None:
    """Clear a single cached instance by key, or everything if key is None."""
    if key is None:
        _CACHE.clear()
    else:
        _CACHE.pop(key, None)


def create_ranker(
    name: str, *, key: Optional[str] = None, refresh: bool = False, **kwargs: Any
) -> Any:
    """Create (or fetch from cache) a ranker by name.

    - name: Registered builder name.
    - key: Optional cache key (defaults to name). Use when same builder is instantiated with different configs.
    - refresh: If True, forces re-creation and replaces any cached instance under the cache key.
    - kwargs: Passed to the registered builder.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Ranker '{name}' is not registered. Registered: {list_registered()}"
        )

    cache_key = key or name
    if not refresh and cache_key in _CACHE:
        return _CACHE[cache_key]

    ranker = _REGISTRY[name](**kwargs)
    _CACHE[cache_key] = ranker
    return ranker


__all__ = [
    "register_ranker",
    "is_registered",
    "list_registered",
    "list_cached",
    "get_cached",
    "clear_cache",
    "create_ranker",
]
