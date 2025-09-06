from __future__ import annotations
import dspy

# -----------------------------
# Prompt 1 — Noun-phrase decomposition (dCS input)
# -----------------------------


class NounPhraseDecompositionSig(dspy.Signature):
    """Decompose the following sentence into individual noun phrases.
    Ignore prefixes such as 'a photo of', 'a picture of', 'a portrait of', etc.
    Your response should ONLY be a list of comma-separated values, e.g.: 'foo, bar, baz'.
    """

    prompt = dspy.InputField(desc="The sentence to decompose into noun phrases.")
    noun_phrases = dspy.OutputField(
        desc="Comma-separated list of noun phrases with no extra text."
    )


class DecomposeNounPhrases(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(NounPhraseDecompositionSig)

    def forward(self, prompt: str) -> str:
        out = self.predict(prompt=prompt)
        return out.noun_phrases.strip()


# -----------------------------
# Prompt 2 — Paraphrasing baseline
# -----------------------------


class ParaphraseBaselineSig(dspy.Signature):
    """Generate {num_solutions} paraphrases of the following image description while
    keeping the semantic meaning of {user_prompt}.
    Respond with each new prompt between <PROMPT> and </PROMPT>, enumerated like:
    1. <PROMPT>paraphrase 1</PROMPT>
    2. <PROMPT>paraphrase 2</PROMPT>
    ... {num_solutions}. <PROMPT>paraphrase {num_solutions}</PROMPT>
    Keep answers concise and effective for text-to-image models.
    """

    user_prompt = dspy.InputField()
    num_solutions = dspy.InputField()
    paraphrases = dspy.OutputField(
        desc="Enumerated list with <PROMPT>...</PROMPT> entries only."
    )


class ParaphraseBaseline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ParaphraseBaselineSig)

    def forward(self, user_prompt: str, num_solutions: int = 4) -> str:
        out = self.predict(user_prompt=user_prompt, num_solutions=str(num_solutions))
        return out.paraphrases.strip()


# -----------------------------
# Prompt 3 — OPT2I with dCS as scorer
# -----------------------------


class OPT2IDCSSig(dspy.Signature):
    """You are an expert prompt optimizer for text-to-image models.
    Your task is to optimize this initial prompt: {user_prompt}.

    Below are previous prompts with a decomposition of their visual elements and a score
    indicating their presence in the generated image. The list is sorted ascending by score (0-100).

    {examples}

    Generate {num_solutions} paraphrases of the initial prompt which keep the semantic meaning
    and have higher scores than ALL the prompts above. Prioritize optimizing for objects with
    the lowest scores. Favor substitutions and reorderings over additions.

    Respond with each new prompt in between <PROMPT> and </PROMPT>, enumerated like:
      1. <PROMPT>paraphrase 1</PROMPT>
      2. <PROMPT>paraphrase 2</PROMPT>
      ... {num_solutions}. <PROMPT>paraphrase {num_solutions}</PROMPT>
    Keep answers concise and effective.
    """

    user_prompt = dspy.InputField()
    num_solutions = dspy.InputField()
    examples = dspy.InputField(
        desc=(
            "Text block of prior prompts with their average scores and per-element CLIP-like scores,"
            " in ascending score order."
        )
    )
    optimized = dspy.OutputField(desc="Enumerated <PROMPT>...</PROMPT> list only.")


class OPT2IwithDCS(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(OPT2IDCSSig)

    def forward(self, user_prompt: str, num_solutions: int, examples: str) -> str:
        out = self.predict(
            user_prompt=user_prompt, num_solutions=str(num_solutions), examples=examples
        )
        return out.optimized.strip()


# -----------------------------
# Prompt 4 — OPT2I with DSG as scorer
# -----------------------------


class OPT2IDSGSig(dspy.Signature):
    """You are an expert prompt optimizer for text-to-image models.
    Your task is to optimize this initial prompt: {user_prompt}.

    Below are previous prompts with the consistency of visual elements in the generated image,
    evaluated via binary questions. The list is sorted ascending by overall consistency score (0-100).

    {examples}

    Generate {num_solutions} paraphrases that keep the semantic meaning and achieve higher scores
    than ALL the prompts above. Focus on visual elements that were inconsistent. Favor substitutions
    and reorderings over additions.

    Respond with each new prompt in between <PROMPT> and </PROMPT>, enumerated like:
      1. <PROMPT>paraphrase 1</PROMPT>
      2. <PROMPT>paraphrase 2</PROMPT>
      ... {num_solutions}. <PROMPT>paraphrase {num_solutions}</PROMPT>
    Keep answers concise and effective.
    """

    user_prompt = dspy.InputField()
    num_solutions = dspy.InputField()
    examples = dspy.InputField(
        desc=(
            "Text block of prior prompts, each with overall DSG score and per-question VQA scores."
        )
    )
    optimized = dspy.OutputField(desc="Enumerated <PROMPT>...</PROMPT> list only.")


class OPT2IwithDSG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(OPT2IDSGSig)

    def forward(self, user_prompt: str, num_solutions: int, examples: str) -> str:
        out = self.predict(
            user_prompt=user_prompt, num_solutions=str(num_solutions), examples=examples
        )
        return out.optimized.strip()


# -----------------------------
# Prompt 5 — Ablations (conciseness variants)
# -----------------------------


class ConcisenessOnlySig(dspy.Signature):
    """Conciseness: Generate {num_solutions} paraphrases of {user_prompt} that
    keep the semantic meaning and achieve higher scores than the examples above (if provided).
    Favor substitutions and reorderings over additions. Respond with enumerated <PROMPT>...</PROMPT> lines only.
    """

    user_prompt = dspy.InputField()
    num_solutions = dspy.InputField()
    examples = dspy.InputField(
        required=False, desc="(Optional) prior prompts and scores."
    )
    outputs = dspy.OutputField(desc="Enumerated <PROMPT>...</PROMPT> list only.")


class ConcisenessOnly(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ConcisenessOnlySig)

    def forward(
        self, user_prompt: str, num_solutions: int = 4, examples: str | None = None
    ) -> str:
        out = self.predict(
            user_prompt=user_prompt,
            num_solutions=str(num_solutions),
            examples=examples or "",
        )
        return out.outputs.strip()


class ConcisenessPrioritizeSig(dspy.Signature):
    """Conciseness + prioritize: Generate {num_solutions} paraphrases of {user_prompt} that
    keep the semantic meaning and have higher scores than the examples above.
    PRIORITIZE optimizing for the objects/elements with the lowest scores in the examples.
    Favor substitutions and reorderings over additions. Respond with enumerated <PROMPT>...</PROMPT>.
    """

    user_prompt = dspy.InputField()
    num_solutions = dspy.InputField()
    examples = dspy.InputField(
        desc="Prior prompts with per-element scores, ascending by score."
    )
    outputs = dspy.OutputField(desc="Enumerated <PROMPT>...</PROMPT> list only.")


class ConcisenessPrioritize(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ConcisenessPrioritizeSig)

    def forward(self, user_prompt: str, num_solutions: int, examples: str) -> str:
        out = self.predict(
            user_prompt=user_prompt, num_solutions=str(num_solutions), examples=examples
        )
        return out.outputs.strip()


class ConcisenessPrioritizeReasoningSig(dspy.Signature):
    """Conciseness + prioritize + reasoning: Briefly reason (max two sentences)
    about why certain objects have higher/lower scores in the examples. Then generate
    {num_solutions} paraphrases of {user_prompt} that keep the semantic meaning, target low-scoring
    objects while preserving high-scoring ones, and are likely to score higher than all examples.
    Favor substitutions and reorderings over additions.
    Respond with up to two reasoning sentences followed by enumerated <PROMPT>...</PROMPT> outputs.
    """

    user_prompt = dspy.InputField()
    num_solutions = dspy.InputField()
    examples = dspy.InputField(
        desc="Prior prompts with element-level scores (ascending)."
    )
    reasoning_and_outputs = dspy.OutputField(
        desc="Short reasoning (≤2 sentences) then enumerated <PROMPT>...</PROMPT> list."
    )


class ConcisenessPrioritizeReasoning(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ConcisenessPrioritizeReasoningSig)

    def forward(self, user_prompt: str, num_solutions: int, examples: str) -> str:
        out = self.predict(
            user_prompt=user_prompt, num_solutions=str(num_solutions), examples=examples
        )
        return out.reasoning_and_outputs.strip()


class ConcisenessPrioritizeStructureSig(dspy.Signature):
    """Conciseness + prioritize + structure: Generate {num_solutions} paraphrases of {user_prompt}
    that keep the semantic meaning and are likely to score higher than all examples.
    PRIORITIZE low-scoring objects while maintaining high-scoring ones.
    FAVOR substitutions and reorderings over additions.
    USE simple words/concepts understandable by text-to-image models (e.g., distinguish foreground vs background).
    Respond with enumerated <PROMPT>...</PROMPT> only.
    """

    user_prompt = dspy.InputField()
    num_solutions = dspy.InputField()
    examples = dspy.InputField(desc="Prior prompts with scores and element breakdowns.")
    outputs = dspy.OutputField(desc="Enumerated <PROMPT>...</PROMPT> list only.")


class ConcisenessPrioritizeStructure(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ConcisenessPrioritizeStructureSig)

    def forward(self, user_prompt: str, num_solutions: int, examples: str) -> str:
        out = self.predict(
            user_prompt=user_prompt, num_solutions=str(num_solutions), examples=examples
        )
        return out.outputs.strip()
