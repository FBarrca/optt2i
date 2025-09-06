# ---------------------------
# Pydantic Models for Structured Outputs
# ---------------------------

from pydantic import BaseModel, Field
from typing import List

__all__ = [
    "_StrictModel",
    "TupleItem",
    "DependencyItem",
    "QuestionItem",
    "TupleOutput",
    "DependencyOutput",
    "QuestionOutput",
]


class _StrictModel(BaseModel):
    # Pydantic v2: forbid extras so the SDK generates a strict JSON Schema
    model_config = dict(extra="forbid")


class TupleItem(_StrictModel):
    """A single tuple item with ID and content."""

    id: int = Field(description="The ID of the tuple")
    content: str = Field(description="The tuple content")


class DependencyItem(_StrictModel):
    """A single dependency item with ID and dependencies."""

    id: int = Field(description="The ID of the tuple")
    dependencies: List[int] = Field(
        description="List of parent tuple IDs this tuple depends on"
    )


class QuestionItem(_StrictModel):
    """A single question item with ID and question text."""

    id: int = Field(description="The ID of the tuple")
    question: str = Field(description="The natural language question for this tuple")


class TupleOutput(_StrictModel):
    """Structured output for tuple generation task."""

    tuples: List[TupleItem] = Field(description="List of generated tuples")


class DependencyOutput(_StrictModel):
    """Structured output for dependency generation task."""

    dependencies: List[DependencyItem] = Field(description="List of tuple dependencies")


class QuestionOutput(_StrictModel):
    """Structured output for question generation task."""

    questions: List[QuestionItem] = Field(description="List of generated questions")
