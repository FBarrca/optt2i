from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from optt2i.ranker.base import Ranker, ScoreOutput, ImageLike
from optt2i.ranker.clipscore import CLIPScoreRanker


# ---------------------------
# Fallback noun-phrase extractor (LLM-backed) used only if DSG pipeline unavailable
# ---------------------------


class _NPResult(BaseModel):
    phrases: List[str]


def _simple_noun_phrases(prompt: str) -> List[str]:
    """Lightweight LLM call to extract noun phrases; used as a fallback.

    Requires OpenAI-compatible environment. If unavailable, returns [].
    """
    try:
        import os
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
        )

        SYSTEM = (
            "You extract concise noun phrases from a sentence. "
            "Ignore prefixes such as 'a photo of', 'a picture of', 'a portrait of'."
        )
        USER = f"""
        Decompose the following sentence into individual noun phrases.
        Return your response as JSON with a 'phrases' key containing a list of strings.

        Here are some examples:

        Prompt: "A ginger cat is sleeping next to the window."
        Output: ginger cat, window

        Prompt: "Many electronic wires pass over the road with few cars on it."
        Output: electronic wires, road, cars

        Prompt: "There is a long hot dog that has toppings on it."
        Output: long hot dog, toppings

        Prompt: "The Mona Lisa wearing a cowboy hat and screaming a punk song into a microphone."
        Output: the mona lisa, cowboy hat, punk song, microphone

        Prompt: "A photograph of a bird wearing headphones and speaking into a microphone in a recording studio."
        Output: bird, headphones, microphone, recording studio

        Prompt: "Concentric squares fading from yellow on the outside to deep orange on the inside."
        Output: concentric squares, yellow, outside, deep orange, inside

        Now process the following prompt:
        {prompt}
        """.strip()

        completion = client.chat.completions.parse(
            model=os.getenv("OPENAI_MODEL"),
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER},
            ],
            response_format=_NPResult,
        )
        return completion.choices[0].message.parsed.phrases
    except Exception:
        return []


# ---------------------------
# Optimized prompt schema (structured generation)
# ---------------------------


class OptimizedPromptItem(BaseModel):
    id: int
    clip: str
    siglip: str


class OptimizedPromptsOutput(BaseModel):
    prompts: List[OptimizedPromptItem]


def _optimize_prompts_with_llm(context: str) -> OptimizedPromptsOutput:
    """Use an OpenAI-compatible LLM to produce per-tuple prompts for CLIP and SigLIP.

    Input context should be original prompt + structured tuples JSON, similar to DSG.
    """
    # Import lazily to avoid hard runtime dependency when LLM is not available.
    from optt2i.ranker.dsg.openai_utils import generate_structured_fn

    PREAMBLE = (
        "Task: given an input caption and a list of skill-specific tuples, "
        "produce short, discriminative text prompts for each tuple optimized for "
        "image–text embedding models. Return JSON with a 'prompts' list of {id, clip, siglip}.\n"
        "Guidelines:\n"
        "- 'clip': concise noun/adjective phrases that maximize CLIP discrimination; avoid stopwords.\n"
        "- 'siglip': natural descriptive caption phrasing that matches the tuple; be explicit.\n"
        "- Do not invent tuples beyond those provided; one prompt per tuple id."
    )

    # Minimal zero-shot instruction using structured outputs
    prompt_text = PREAMBLE + "\n\nInput:\n" + context + "\n\nOutput:"
    return generate_structured_fn(prompt_text, OptimizedPromptsOutput)


class DecomposedCLIPScoreRanker(Ranker):
    """DSG-style decomposition + per-tuple optimized prompts for CLIP and SigLIP.

    Behavior:
    - If an OpenAI-compatible LLM is available, decompose the caption into DSG tuples,
      then generate per-tuple optimized prompts for CLIP and SigLIP, score each, and aggregate.
    - If LLM/DSG is unavailable, fall back to LLM noun-phrase extraction (if possible)
      and average per-phrase CLIPScores.

    Parameters
    - clip_model_name: HF id for the CLIP model to use (default: openai/clip-vit-base-patch32)
    - siglip_model_name: HF id for SigLIP (default: from defaults.py)
    - phrase_extractor: Optional callable(prompt)->List[str] used only as fallback.
    """

    name = "decomposed_clipscore"

    def __init__(
        self,
        *,
        clip_model_name: Optional[str] = "openai/clip-vit-base-patch32",
        siglip_model_name: Optional[str] = None,
        phrase_extractor: Optional[Any] = None,
        clip_ranker: Optional[CLIPScoreRanker] = None,
        **kwargs: Any,
    ) -> None:
        # Two backends: one CLIP, one SigLIP
        from optt2i.ranker.defaults import DEFAULT_CLIP_MODEL as DEFAULT_SIGLIP

        self.clip_ranker_clip = CLIPScoreRanker(model_name=clip_model_name)
        # If a prebuilt ranker is provided, prefer it as the SigLIP backend for continuity
        self.clip_ranker_siglip = clip_ranker or CLIPScoreRanker(
            model_name=(siglip_model_name or DEFAULT_SIGLIP)
        )
        # Fallback concept extractor
        self.phrase_extractor = phrase_extractor or _simple_noun_phrases

    def _score_with_fallback(self, prompt: str, image: ImageLike) -> ScoreOutput:
        phrases = self.phrase_extractor(prompt) or [prompt]

        per_phrase: List[Dict[str, Any]] = []
        scores: List[float] = []
        for ph in phrases:
            r = self.clip_ranker_siglip.score(ph, image)
            per_phrase.append({"concept": ph, "siglip_score": r.global_score})
            scores.append(r.global_score)

        global_score = float(sum(scores) / max(1, len(scores)))
        details = {
            "mode": "fallback_noun_phrases",
            "concepts": per_phrase,
            "num_concepts": len(phrases),
        }
        return ScoreOutput(global_score=global_score, details=details)

    def score(self, prompt: str, image: ImageLike, **kwargs: Any) -> ScoreOutput:
        # Try DSG-style tuples + optimized prompts. Import lazily to avoid hard dependency.
        try:
            from optt2i.ranker.dsg.generator import generate_dsg_structured
        except Exception:
            return self._score_with_fallback(prompt, image)

        try:
            tuple_outputs, _question_outputs_unused, dependency_outputs = (
                generate_dsg_structured([prompt], verbose=False)
            )
            tuple_out = tuple_outputs[0]
            dep_out = dependency_outputs[0]
            # Build the same context as DSG does: caption + structured tuples JSON
            context = "\n".join([prompt, str(tuple_out.model_dump())])

            opt = _optimize_prompts_with_llm(context)
            items = opt.prompts or []
            if not items:
                # If optimizer returned nothing, fall back to using tuple content as prompts
                items = [
                    OptimizedPromptItem(id=t.id, clip=t.content, siglip=t.content)
                    for t in tuple_out.tuples
                ]

            per_tuple: List[Dict[str, Any]] = []
            clip_scores: List[float] = []
            siglip_scores: List[float] = []
            for it in items:
                r_clip = self.clip_ranker_clip.score(it.clip, image)
                r_sig = self.clip_ranker_siglip.score(it.siglip, image)
                per_tuple.append(
                    {
                        "id": it.id,
                        "clip_prompt": it.clip,
                        "clip_score": r_clip.global_score,
                        "siglip_prompt": it.siglip,
                        "siglip_score": r_sig.global_score,
                    }
                )
                clip_scores.append(r_clip.global_score)
                siglip_scores.append(r_sig.global_score)

            clip_global = float(sum(clip_scores) / max(1, len(clip_scores)))
            siglip_global = float(sum(siglip_scores) / max(1, len(siglip_scores)))

            details: Dict[str, Any] = {
                "mode": "dsg_optimized",
                "tuples": [t.model_dump() for t in tuple_out.tuples],
                "dependencies": dep_out,
                "per_tuple": per_tuple,
                "aggregates": {
                    "clip_global": clip_global,
                    "siglip_global": siglip_global,
                },
            }
            # Use SigLIP aggregate as the main scalar (defaults use SigLIP)
            return ScoreOutput(global_score=siglip_global, details=details)
        except Exception:
            # Any failure: revert to simple noun-phrase fallback
            return self._score_with_fallback(prompt, image)


__all__ = [
    "DecomposedCLIPScoreRanker",
]


if __name__ == "__main__":
    # Minimal sanity run (structure only). Heavy plotting removed.
    ranker = DecomposedCLIPScoreRanker()
    demo_prompts = [
        "A small green tank in a field from very far away driving towards a house in a forest.",
        "A ginger cat sleeping by a sunlit window.",
    ]
    demo_images = [
        "qwen_demo.png",
    ]
    for img in demo_images:
        for pr in demo_prompts:
            try:
                out = ranker.score(pr, img)
                print("-", img, "|", pr)
                print("  global:", out.global_score)
                print("  mode:", out.details.get("mode"))
            except Exception as e:
                print("ERROR:", e)
