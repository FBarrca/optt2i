from __future__ import annotations

from typing import Any, Dict, Optional, Mapping, Iterable, List

from optt2i.ranker.base import Ranker, ScoreOutput, ImageLike
from optt2i.ranker.dsg.vqa import OpenAIVisionVQA


class DSGScore:
    def __init__(
        self,
        dependency_output: Mapping[str, Any],
        question_output: Mapping[str, Any],
        *,
        vqa: Optional[Any] = None,
    ) -> None:
        # Allow custom VQA implementation injection for testing or alternate backends.
        # Expected to expose .batch_vqa(image, questions, binary=True) -> List[str]
        self.vqa_model = vqa or OpenAIVisionVQA()

        # dependency_graph: {child_id: [parent_id, ...]}
        self.dependency_graph = {
            d["id"]: list(d.get("dependencies", []))
            for d in dependency_output.get("dependencies", [])
        }

        # Keep questions keyed by ID and preserve the input order of IDs
        q_items = question_output.get("questions", [])
        self.questions_by_id = {q["id"]: q["question"] for q in q_items}
        self.question_ids = [q["id"] for q in q_items]  # preserve order as provided
        self.id_to_index = {qid: i for i, qid in enumerate(self.question_ids)}

        # This is the ordered list we actually ask the VQA model
        self.questions = [self.questions_by_id[qid] for qid in self.question_ids]

    def score(self, image: Any) -> float:
        # Get raw answers aligned with self.questions (i.e., self.question_ids order)
        raw_answers: List[str] = self.vqa_model.batch_vqa(image, self.questions)

        # Map answers back to IDs
        answers_by_id = {
            qid: raw_answers[self.id_to_index[qid]] for qid in self.question_ids
        }

        # Apply dependency propagation by ID:
        # if any parent is 'no', force the child to 'no'
        for child_id, parent_ids in self.dependency_graph.items():
            if child_id not in answers_by_id:
                continue  # child wasn't asked (defensive)
            for pid in parent_ids:
                if answers_by_id.get(pid) == "no":
                    answers_by_id[child_id] = "no"
                    break  # no need to check other parents

        # Print once per question, after dependency adjustment
        # No stdout noise here; callers can inspect details if needed.

        # Score
        total = len(answers_by_id) or 1
        yes_count = sum(1 for a in answers_by_id.values() if a == "yes")
        return yes_count / total


if __name__ == "__main__":
    # Simple manual test; requires network and valid OpenAI env to run fully.
    from optt2i.ranker.dsg.generator import generate_dsg_structured

    prompt = "A small green tank in a field from very far away driving towards a house in a forest."
    tuple_outputs, question_outputs, dependency_outputs = generate_dsg_structured(
        [prompt]
    )
    dsg_score = DSGScore(dependency_outputs[0], question_outputs[0].model_dump())
    # Example paths; replace with local images for manual testing
    print(dsg_score.score("image.png"))
    print(dsg_score.score("good.png"))


class DSGRanker(Ranker):
    """DSG (Davidsonian Scene Graph) ranker that uses structured generation and VQA.

    This ranker generates structured scene graphs from prompts and uses VQA to score
    the consistency between the generated structure and the actual image.

    Parameters:
    - vqa_ranker: Optional VQA ranker to use for scoring (defaults to OpenAIVisionVQA)
    """

    name = "dsg"

    def __init__(
        self,
        vqa_ranker: Optional[Any] = None,
        question_generator: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        # Optional external VQA object compatible with OpenAIVisionVQA API
        # (exposes .batch_vqa(image, questions, binary=True) -> List[str]).
        self.vqa_ranker = vqa_ranker

        # Optional callable: question_generator(prompt) -> Mapping with "questions" key
        # Each item: {"id": Any, "question": str, (optional) "depends_on": List[Any]}
        self.question_generator = question_generator

        # Cache last prompt-specific DSGScore
        self._dsg_score: Optional[DSGScore] = None
        self._cached_prompt: Optional[str] = None

    def _ensure_dsg_score(self, prompt: str) -> DSGScore:
        """Generate DSG structure for the given prompt, with prompt-aware caching."""
        if self._dsg_score is not None and self._cached_prompt == prompt:
            return self._dsg_score

        # If a lightweight question generator is provided, use it.
        if self.question_generator is not None:
            gen = self.question_generator(prompt)
            # Normalize to expected shapes
            questions_iter: Iterable[Mapping[str, Any]] = list(gen.get("questions", []))
            question_output: Dict[str, Any] = {
                "questions": [
                    {"id": q.get("id"), "question": q.get("question")}
                    for q in questions_iter
                    if q.get("question") is not None
                ]
            }
            dependencies_list: List[Dict[str, Any]] = []
            for q in questions_iter:
                deps = list(q.get("depends_on", []) or [])
                if deps:
                    dependencies_list.append({"id": q.get("id"), "dependencies": deps})
            dependency_output: Dict[str, Any] = {"dependencies": dependencies_list}

            self._dsg_score = DSGScore(
                dependency_output,
                question_output,
                vqa=self.vqa_ranker,
            )
            self._cached_prompt = prompt
            return self._dsg_score

        # Otherwise, fall back to the structured multi-step pipeline
        from .generator import generate_dsg_structured

        tuple_outputs, question_outputs, dependency_outputs = generate_dsg_structured(
            [prompt]
        )

        self._dsg_score = DSGScore(
            dependency_outputs[0], question_outputs[0].model_dump(), vqa=self.vqa_ranker
        )
        self._cached_prompt = prompt
        return self._dsg_score

    def score(self, prompt: str, image: ImageLike, **kwargs: Any) -> ScoreOutput:
        """Score the consistency between a prompt and image using DSG.

        Args:
            prompt: Text description to evaluate
            image: Image to evaluate against
            **kwargs: Additional arguments (unused)

        Returns:
            ScoreOutput with global_score and details
        """
        dsg_score = self._ensure_dsg_score(prompt)

        # Get the score using the DSGScore instance
        global_score = dsg_score.score(image)

        details: Dict[str, Any] = {
            "dsg_score": global_score,
            "vqa_ranker": self.vqa_ranker,
            "prompt": prompt,
        }

        return ScoreOutput(global_score=global_score, details=details)
