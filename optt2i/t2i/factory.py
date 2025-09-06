"""Registry and cache for text-to-image (t2i) models.

This module provides a minimal factory with in-memory caching for t2i models
such as Hugging Face diffusers pipelines or Nunchaku-backed Qwen-Image.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

_REGISTRY: Dict[str, Callable[..., Any]] = {}
_CACHE: Dict[str, Any] = {}


def register_model(name: str, builder: Callable[..., Any]) -> None:
    """Register a model builder by name."""
    if not isinstance(name, str) or not name:
        raise ValueError("Model name must be a non-empty string")
    _REGISTRY[name] = builder


def is_registered(name: str) -> bool:
    return name in _REGISTRY


def list_registered() -> List[str]:
    return sorted(_REGISTRY.keys())


def list_cached() -> List[str]:
    return sorted(_CACHE.keys())


def get_cached(key: str) -> Any:
    if key not in _CACHE:
        raise KeyError(f"No cached model under key '{key}'")
    return _CACHE[key]


def clear_cache(key: Optional[str] = None) -> None:
    """Clear a single cached instance by key, or everything if key is None."""
    if key is None:
        _CACHE.clear()
    else:
        _CACHE.pop(key, None)


def create_model(
    name: str, *, key: Optional[str] = None, refresh: bool = False, **kwargs: Any
) -> Any:
    """Create (or fetch from cache) a t2i model by name.

    - name: Registered builder name.
    - key: Optional cache key (defaults to name). Use when same builder is instantiated with different configs.
    - refresh: If True, forces re-creation and replaces any cached instance under the cache key.
    - kwargs: Passed to the registered builder.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Model '{name}' is not registered. Registered: {list_registered()}"
        )

    cache_key = key or name
    if not refresh and cache_key in _CACHE:
        return _CACHE[cache_key]

    model = _REGISTRY[name](**kwargs)
    _CACHE[cache_key] = model
    return model


__all__ = [
    "register_model",
    "is_registered",
    "list_registered",
    "list_cached",
    "get_cached",
    "clear_cache",
    "create_model",
]
