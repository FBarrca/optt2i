from __future__ import annotations

"""Quick test driver for rankers.

Examples:
  python -m optt2i.ranker.main \
    --images ./qwen_demo.png \
    --prompts "a bike lying on the ground, covered in snow" \
    --methods clipscore,decomposed_clipscore

  python -m optt2i.ranker.main --images img1.png img2.png --prompts "prompt 1" "prompt 2" --methods clipscore,dsg
"""

import argparse
import os
import sys
from typing import Iterable, List, Sequence

from optt2i.ranker import score_with_methods
from optt2i.ranker.llm import (
    get_openai_client,
    llm_noun_phrase_extractor,
    llm_dsg_question_generator,
)


DEFAULT_PROMPTS = [
    "a closeup of a tank",
    "a tank from very far away",
    "a tank from far away",
    "a cat in a desert",
    "a car in a desert",
]


def _default_images() -> List[str]:
    here = os.getcwd()
    candidates = [
        os.path.join(here, "qwen_demo.png"),
    ]
    return [p for p in candidates if os.path.isfile(p)]


def _resolve_pairs(
    images: Sequence[str], prompts: Sequence[str]
) -> List[tuple[str, str]]:
    pairs: List[tuple[str, str]] = []
    if not images:
        raise SystemExit("No images provided and no default image found.")
    if not prompts:
        raise SystemExit("No prompts provided.")

    if len(images) == len(prompts):
        pairs = list(zip(images, prompts))
    elif len(prompts) == 1:
        pairs = [(img, prompts[0]) for img in images]
    elif len(images) == 1:
        pairs = [(images[0], pr) for pr in prompts]
    else:
        # Cartesian product if both > 1 and lengths differ
        pairs = [(img, pr) for img in images for pr in prompts]
    return pairs


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test rankers with images and prompts")
    p.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="Image paths (defaults to ./qwen_demo.png if present)",
    )
    p.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="Prompts (defaults to two sample prompts)",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="clipscore,decomposed_clipscore",
        help="Comma-separated methods: clipscore,decomposed_clipscore,vqa,vqa_instructblip,dsg",
    )
    # LLM options (OpenAI-compatible)
    p.add_argument(
        "--llm-np",
        action="store_true",
        help="Use LLM for noun-phrase extraction (decomposed_clipscore)",
    )
    p.add_argument(
        "--llm-dsg",
        action="store_true",
        help="Use LLM for DSG question generation (dsg)",
    )
    p.add_argument(
        "--llm-base", type=str, default=None, help="OpenAI-compatible API base URL"
    )
    p.add_argument(
        "--llm-key", type=str, default=None, help="OpenAI-compatible API key"
    )
    p.add_argument(
        "--llm-model", type=str, default=None, help="LLM model name (e.g., gpt-4o-mini)"
    )
    # No path needed with openai client
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    ns = parse_args(argv or sys.argv[1:])

    images = (
        ns.images if ns.images is not None and len(ns.images) > 0 else _default_images()
    )
    prompts = (
        ns.prompts
        if ns.prompts is not None and len(ns.prompts) > 0
        else DEFAULT_PROMPTS
    )
    methods = [m.strip() for m in str(ns.methods).split(",") if m.strip()]

    pairs = _resolve_pairs(images, prompts)

    # Optional LLM client
    client = None
    if ns.llm_np or ns.llm_dsg or ns.llm_base or ns.llm_key or ns.llm_model:
        try:
            client = get_openai_client(api_base=ns.llm_base, api_key=ns.llm_key)
        except Exception as e:
            print(f"Warning: could not initialize LLM client: {e}")
            client = None

    print(f"Methods: {methods}")
    for idx, (img, pr) in enumerate(pairs, start=1):
        print(f"\n[{idx}] Image: {img}")
        print(f"Prompt: {pr}")
        # Optional per-method params
        ranker_params = {}
        if client is not None and ns.llm_np and "decomposed_clipscore" in methods:
            ranker_params["decomposed_clipscore"] = {
                "phrase_extractor": llm_noun_phrase_extractor(
                    client, model=ns.llm_model
                )
            }
        if client is not None and ns.llm_dsg and "dsg" in methods:
            ranker_params["dsg"] = {
                "question_generator": llm_dsg_question_generator(
                    client, model=ns.llm_model
                )
            }

        try:
            outputs = score_with_methods(pr, img, methods, ranker_params=ranker_params)
        except Exception as e:
            print(f"  ERROR while scoring: {e}")
            continue
        for name, out in outputs.items():
            print(f"  - {name}: {out.global_score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
