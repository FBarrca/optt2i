from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore

try:
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    OpenAI = None  # type: ignore


def get_openai_client(
    *,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    organization: Optional[str] = None,
    project: Optional[str] = None,
):
    """Create an OpenAI client, supporting OpenAI-compatible base URLs.

    Env fallbacks: OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_ORG, OPENAI_PROJECT.
    """
    if OpenAI is None:
        raise RuntimeError(
            "The 'openai' package is not installed. Please install it to enable LLM features."
        )

    # Load environment from .env if available
    if load_dotenv is not None:
        try:
            load_dotenv()
        except Exception:
            pass

    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPTT2I_LLM_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY (or OPTT2I_LLM_API_KEY) is required for LLM features."
        )

    base_url = api_base or os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
    org = organization or os.getenv("OPENAI_ORG")
    proj = project or os.getenv("OPENAI_PROJECT")

    # The OpenAI client supports base_url kwarg for compatible endpoints
    client = OpenAI(api_key=key, base_url=base_url, organization=org, project=proj)
    return client


def _chat_json(
    client, *, model: Optional[str], system: str, user: str
) -> Dict[str, Any]:
    """Call chat.completions with JSON output enforced.

    Assumes an OpenAI-compatible endpoint that honors JSON responses.
    """
    # Resolve model from env if not provided
    model = (
        model
        or os.getenv("OPENAI_MODEL")
        or os.getenv("OPTT2I_LLM_MODEL")
        or "gpt-4o-mini"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    return json.loads(content)


def llm_noun_phrase_extractor(client, *, model: Optional[str] = None):
    """Return a callable(prompt)->List[str] using an OpenAI-compatible LLM."""

    def _extract(prompt: str) -> List[str]:
        system = (
            "Extract short, distinct noun phrases relevant for scoring prompt–image consistency. "
            'Return only JSON: {"noun_phrases": ["phrase1", ...]}.'
        )
        user = (
            "Prompt: " + prompt + "\n\n"
            "Rules: lowercase, 1–4 words, deduplicated, order by salience."
        )
        try:
            obj = _chat_json(client, model=model, system=system, user=user)
            phrases = obj.get("noun_phrases", [])
            phrases = [str(p).strip().lower() for p in phrases if str(p).strip()]
            # Dedup preserving order
            seen = set()
            out: List[str] = []
            for p in phrases:
                if p not in seen:
                    seen.add(p)
                    out.append(p)
            return out
        except Exception:
            return []

    return _extract


def llm_dsg_question_generator(client, *, model: Optional[str] = None):
    """Return a callable(prompt)->graph dict powered by an OpenAI-compatible LLM.

    Output schema:
    {
      "nodes": ["bike", ...],
      "edges": [{"from": "bike", "to": "ground", "relation": "on"}, ...],
      "questions": [
        {"id": "exists:bike", "question": "Is there a bike?", "depends_on": []},
        {"id": "rel:bike:on:ground", "question": "Is the bike on the ground?", "depends_on": ["exists:bike"]}
      ]
    }
    """

    def _generate(prompt: str) -> Dict[str, Any]:
        system = (
            "Decompose a caption into atomic, unique yes/no questions with dependencies for visual QA. "
            "Return only JSON with keys nodes (entities), edges (from,to,relation), questions (id,question,depends_on[])."
        )
        user = (
            "Caption: " + prompt + "\n\n"
            "Include existence questions for key entities and relation questions for 'on', 'in', 'with', 'covered in', 'lying on'. "
            "Relation questions must depend on the existence of the subject entity."
        )
        try:
            obj = _chat_json(client, model=model, system=system, user=user)
            nodes = list(obj.get("nodes", []))
            edges = list(obj.get("edges", []))
            questions = list(obj.get("questions", []))
            return {"nodes": nodes, "edges": edges, "questions": questions}
        except Exception:
            return {"nodes": [], "edges": [], "questions": []}

    return _generate


__all__ = [
    "get_openai_client",
    "llm_noun_phrase_extractor",
    "llm_dsg_question_generator",
]
