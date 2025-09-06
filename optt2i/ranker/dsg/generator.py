from __future__ import annotations

import string
from typing import List, Type, Callable

import tqdm
from pydantic import BaseModel

from optt2i.ranker.dsg.utils import (
    TIFA160_ICL_TRAIN_IDS,
    get_tifa_examples,
    load_tifa160_data,
)
from optt2i.ranker.dsg.types import DependencyOutput, QuestionOutput, TupleOutput
from optt2i.ranker.dsg.openai_utils import generate_structured_fn

# ---------------------------
# Training examples (preload once)
# ---------------------------

_TIFA160_DF = load_tifa160_data()
_TUPLE_EXAMPLES = get_tifa_examples(_TIFA160_DF, TIFA160_ICL_TRAIN_IDS, task="tuple")
_DEPENDENCY_EXAMPLES = get_tifa_examples(
    _TIFA160_DF, TIFA160_ICL_TRAIN_IDS, task="dependency"
)
_QUESTION_EXAMPLES = get_tifa_examples(
    _TIFA160_DF, TIFA160_ICL_TRAIN_IDS, task="question"
)

# ---------------------------
# Prompt templates
# ---------------------------

_PROMPT_TEMPLATE = string.Template(
    """
$preamble

$examples

$test_input_output
""".strip()
)

_EXAMPLES_TEMPLATE = string.Template(
    """
$input_name: $input
$output_name: $output""".strip()
)

_TEST_TEMPLATE = string.Template(
    """
$input_name: $test_input
$output_name: """.lstrip()
)

_TUPLE_PREAMBLE = (
    "Task: given input prompts, describe each scene with skill-specific tuples.\n"
    "Do not duplicate tuples. Do not add tuples that are not explicitly supported by the prompt.\n"
    "Return a JSON object matching the schema with a `tuples` list of {id, content}."
)
_DEPENDENCY_PREAMBLE = (
    "Task: given input prompts and tuples, list parent tuples for each tuple.\n"
    "Return a JSON object with a `dependencies` list of {id, dependencies} where dependencies is an array of parent IDs."
)
_QUESTION_PREAMBLE = (
    "Task: given input prompts and skill-specific tuples, rewrite each tuple as a natural language question.\n"
    "Return a JSON object with a `questions` list of {id, question}."
)

# ---------------------------
# Prompt assembly
# ---------------------------


def make_prompt(
    examples: List[dict],
    test_input: str,
    preamble: str,
    input_name: str = "input",
    output_name: str = "output",
) -> str:
    examples_str = "\n\n".join(
        _EXAMPLES_TEMPLATE.substitute(
            input_name=input_name,
            output_name=output_name,
            input=ex["input"].strip(),
            output=ex["output"].strip(),
        )
        for ex in examples
    )
    test_input_str = _TEST_TEMPLATE.substitute(
        input_name=input_name, output_name=output_name, test_input=test_input
    )
    return _PROMPT_TEMPLATE.substitute(
        preamble=preamble, examples=examples_str, test_input_output=test_input_str
    )


# ---------------------------
# Helpers
# ---------------------------

_EMPTY_OUTPUT_FACTORY: dict[Type[BaseModel], Callable[[], BaseModel]] = {
    TupleOutput: lambda: TupleOutput(tuples=[]),
    DependencyOutput: lambda: DependencyOutput(dependencies=[]),
    QuestionOutput: lambda: QuestionOutput(questions=[]),
}


def _empty_output_for(model: Type[BaseModel]) -> BaseModel:
    if model not in _EMPTY_OUTPUT_FACTORY:
        raise ValueError(f"No empty-output factory registered for {model!r}")
    return _EMPTY_OUTPUT_FACTORY[model]()


def _build_context(prompt: str, tuple_output: TupleOutput) -> str:
    return "\n".join([prompt, str(tuple_output.model_dump())])


def _sanitize_dependencies_inplace(dep_dicts: List[dict]) -> None:
    for item in dep_dicts:
        for dep in item.get("dependencies", []):
            dep["dependencies"] = [d for d in dep.get("dependencies", []) if d != 0]


# ---------------------------
# Generation (structured only) — list in, list out
# ---------------------------


def generate_structured_with_in_context_examples(
    inputs: List[str],
    train_examples: List[dict],
    preamble: str,
    output_model: Type[BaseModel],
    input_name: str = "input",
    output_name: str = "output",
    verbose: bool = True,
) -> List[BaseModel]:
    """Generate structured outputs for a list of inputs using few-shot prompts."""
    prompts = []
    for x in tqdm.tqdm(
        range(len(inputs)), desc="Preparing LM inputs", disable=not verbose
    ):
        prompts.append(
            make_prompt(train_examples, inputs[x], preamble, input_name, output_name)
        )

    outputs: List[BaseModel] = []
    for i in tqdm.tqdm(
        range(len(prompts)), desc="Running structured LM calls", disable=not verbose
    ):
        try:
            outputs.append(generate_structured_fn(prompts[i], output_model))
        except Exception as e:
            if verbose:
                print(f"Error processing item {i}: {e}")
            outputs.append(_empty_output_for(output_model))
    return outputs


# ---------------------------
# Full DSG pipeline (structured only) — list API
# ---------------------------


def generate_dsg_structured(
    prompts: List[str],
    tuple_train_examples=_TUPLE_EXAMPLES,
    dependency_train_examples=_DEPENDENCY_EXAMPLES,
    question_train_examples=_QUESTION_EXAMPLES,
    verbose: bool = True,
) -> tuple[List[TupleOutput], List[QuestionOutput], List[dict]]:
    """Run the full pipeline over a list of prompts.

    Returns three lists aligned with the input prompts:
      - tuple_outputs:  List[TupleOutput]
      - question_outputs: List[QuestionOutput]
      - dependency_outputs: List[dict] (model_dump(), with ID==0 refs removed)
    """
    # 1) Tuples
    tuple_outputs: List[TupleOutput] = generate_structured_with_in_context_examples(
        prompts, tuple_train_examples, _TUPLE_PREAMBLE, TupleOutput, verbose=verbose
    )  # type: ignore

    # 2) Questions (context = prompt + tuples)
    contexts = [_build_context(p, t) for p, t in zip(prompts, tuple_outputs)]
    question_outputs: List[QuestionOutput] = (
        generate_structured_with_in_context_examples(
            contexts,
            question_train_examples,
            _QUESTION_PREAMBLE,
            QuestionOutput,
            verbose=verbose,
        )
    )  # type: ignore

    # 3) Dependencies (same context)
    dependency_models = generate_structured_with_in_context_examples(
        contexts,
        dependency_train_examples,
        _DEPENDENCY_PREAMBLE,
        DependencyOutput,
        verbose=verbose,
    )  # type: ignore

    dependency_outputs: List[dict] = [m.model_dump() for m in dependency_models]
    _sanitize_dependencies_inplace(dependency_outputs)

    return tuple_outputs, question_outputs, dependency_outputs


# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    prompts = [
        "A small green tank in a field from very far away driving towards a house in a forest."
    ]

    tuples, questions, dependencies = generate_dsg_structured(
        prompts,
        tuple_train_examples=_TUPLE_EXAMPLES,
        dependency_train_examples=_DEPENDENCY_EXAMPLES,
        question_train_examples=_QUESTION_EXAMPLES,
        verbose=True,
    )

    from pprint import pprint

    # Show results for the first prompt
    pprint(tuples[0].model_dump())
    pprint(questions[0].model_dump())
    pprint(dependencies[0])

    from optt2i.ranker.dsg.render_dsg_dag import render_dag_pygraphviz

    render_dag_pygraphviz(
        tuples[0].model_dump(),
        questions[0].model_dump(),
        dependencies[0],
        "demo.png",
    )
