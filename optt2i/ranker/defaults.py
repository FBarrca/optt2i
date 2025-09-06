"""Default model names and configs for rankers.

These are chosen for broad availability on the Hugging Face Hub. Models are
only referenced by name; ensure weights are available locally if running
offline.
"""

# CLIP thinks in comparisons ("this caption vs. that caption for this image").
# SigLIP thinks in matches ("does this caption match this image, yes or no?").

# CLIP text-image model
DEFAULT_CLIP_MODEL = "google/siglip-so400m-patch14-384"

# VQA model; ViLT and BLIP are common choices
DEFAULT_VQA_PIPELINE_MODEL = "dandelin/vilt-b32-finetuned-vqa"

__all__ = [
    "DEFAULT_CLIP_MODEL",
    "DEFAULT_VQA_PIPELINE_MODEL",
]
