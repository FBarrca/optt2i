import os
from typing import Any, Dict, List

import pytest

from optt2i.ranker import create_ranker, list_registered
from optt2i.ranker.base import ScoreOutput
from optt2i.ranker.clipscore import CLIPScoreRanker
from optt2i.ranker.vqa import VQARanker
from optt2i.ranker.instruct_blip_vqa import InstructBLIPVQARanker


def _sample_image_path() -> str:
    # Use a small local image shipped with the repo
    here = os.getcwd()
    for cand in [
        os.path.join(here, "image.png"),
        os.path.join(here, "good.png"),
        os.path.join(here, "demo.png"),
    ]:
        if os.path.isfile(cand):
            return cand
    # As a last resort, any missing path will be caught by PIL with a clear error
    return "image.png"


def test_registered_rankers_present():
    names = set(list_registered())
    expected = {
        "clipscore",
        "decomposed_clipscore",
        "vqa",
        "vqa_instructblip",
        "dsg",
        "dsg_structured",
    }
    assert expected.issubset(names)


def test_clipscore_factory_score_mocked(monkeypatch):
    def fake_score(self, prompt: str, image: Any, **kwargs: Any) -> ScoreOutput:
        return ScoreOutput(global_score=0.42, details={"mock": True})

    monkeypatch.setattr(CLIPScoreRanker, "score", fake_score, raising=True)
    ranker = create_ranker("clipscore", refresh=True)
    out = ranker.score("a test prompt", _sample_image_path())
    assert isinstance(out, ScoreOutput)
    assert 0.0 <= out.global_score <= 1.0
    assert out.details.get("mock") is True


def test_decomposed_clipscore_factory_score_dsg_style(monkeypatch):
    # 1) Stub CLIP scoring to avoid heavy models
    def fake_score(self, prompt: str, image: Any, **kwargs: Any) -> ScoreOutput:
        # Return deterministic score regardless of backend
        return ScoreOutput(global_score=0.8, details={"prompt": prompt})

    monkeypatch.setattr(CLIPScoreRanker, "score", fake_score, raising=True)

    # 2) Stub DSG structured generation to produce tuples/dep
    from optt2i.ranker.dsg.types import (
        TupleItem,
        TupleOutput,
        QuestionOutput,
    )

    def fake_generate_dsg_structured(prompts: List[str], **kwargs: Any):
        tuple_outs = [
            TupleOutput(tuples=[TupleItem(id=1, content="bike"), TupleItem(id=2, content="ground")])
            for _ in prompts
        ]
        question_outs = [QuestionOutput(questions=[]) for _ in prompts]
        dep_outs = [{"dependencies": []} for _ in prompts]
        return tuple_outs, question_outs, dep_outs

    monkeypatch.setattr(
        "optt2i.ranker.dsg.generator.generate_dsg_structured",
        fake_generate_dsg_structured,
        raising=True,
    )

    # 3) Stub optimized prompt generator to produce per-tuple clip/siglip prompts
    from optt2i.ranker.decomposed_clipscore import (
        OptimizedPromptsOutput,
        OptimizedPromptItem,
    )

    def fake_optimize(context: str) -> OptimizedPromptsOutput:
        return OptimizedPromptsOutput(
            prompts=[
                OptimizedPromptItem(id=1, clip="bike", siglip="a bike"),
                OptimizedPromptItem(id=2, clip="ground", siglip="on the ground"),
            ]
        )

    monkeypatch.setattr(
        "optt2i.ranker.decomposed_clipscore._optimize_prompts_with_llm",
        fake_optimize,
        raising=True,
    )

    ranker = create_ranker("decomposed_clipscore", refresh=True)
    out = ranker.score("a bike on the ground", _sample_image_path())
    assert isinstance(out, ScoreOutput)
    assert out.details.get("mode") == "dsg_optimized"
    assert "aggregates" in out.details
    assert 0.0 <= out.global_score <= 1.0


def test_vqa_factory_score_mocked(monkeypatch):
    # Replace the HF pipeline with a trivial callable returning yes
    def fake_ensure_pipeline(self):
        def _qa(image=None, question: str = ""):
            return {"answer": "yes", "score": 0.9}

        return _qa

    monkeypatch.setattr(VQARanker, "_ensure_pipeline", fake_ensure_pipeline, raising=True)
    ranker = create_ranker("vqa", refresh=True)
    out = ranker.score("Is there a bike?", _sample_image_path())
    assert isinstance(out, ScoreOutput)
    assert out.global_score >= 0.0
    assert out.details.get("qa")


def test_dsg_factory_score_with_injected_dependencies(monkeypatch):
    # Inject a simple question generator and VQA backend
    def qgen(prompt: str) -> Dict[str, Any]:
        return {
            "questions": [
                {"id": 1, "question": "Is there a bike?", "depends_on": []},
                {"id": 2, "question": "Is the bike on the ground?", "depends_on": [1]},
            ]
        }

    class DummyVQA:
        def batch_vqa(self, image, questions, binary=True):
            return ["yes" for _ in questions]

    ranker = create_ranker(
        "dsg", refresh=True, vqa_ranker=DummyVQA(), question_generator=qgen
    )
    out = ranker.score("a bike on the ground", _sample_image_path())
    assert isinstance(out, ScoreOutput)
    assert 0.0 <= out.global_score <= 1.0
    assert out.details.get("prompt")


def test_vqa_instructblip_factory_score_mocked(monkeypatch):
    def fake_answer_one(self, image, question: str):
        return {"answer": "yes", "score": 1.0}

    monkeypatch.setattr(
        InstructBLIPVQARanker, "_answer_one", fake_answer_one, raising=True
    )
    ranker = create_ranker("vqa_instructblip", refresh=True)
    out = ranker.score("Is there a bike?", _sample_image_path())
    assert isinstance(out, ScoreOutput)
    assert out.global_score == 1.0


def test_dsg_structured_alias_builds(monkeypatch):
    # Ensure alias builder is registered and can instantiate with injection
    class DummyVQA:
        def batch_vqa(self, image, questions, binary=True):
            return ["yes" for _ in questions]

    def qgen(prompt: str) -> Dict[str, Any]:
        return {"questions": [{"id": 1, "question": "Is there a cat?"}]}

    ranker = create_ranker(
        "dsg_structured", refresh=True, vqa_ranker=DummyVQA(), question_generator=qgen
    )
    out = ranker.score("a cat", _sample_image_path())
    assert isinstance(out, ScoreOutput)
    assert 0.0 <= out.global_score <= 1.0

