"""Text-to-image factory public API.

The implementation lives in submodules under this package. This file only
re-exports the public interface and registers default builders.
"""

from .factory import (
    register_model,
    is_registered,
    list_registered,
    list_cached,
    get_cached,
    clear_cache,
    create_model,
)
from .defaults import register_defaults

# Register default builders (Hugging Face diffusers + Qwen-Image Lightning)
register_defaults()

__all__ = [
    "register_model",
    "is_registered",
    "list_registered",
    "list_cached",
    "get_cached",
    "clear_cache",
    "create_model",
    "register_defaults",
]
