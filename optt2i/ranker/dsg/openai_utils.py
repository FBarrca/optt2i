import os
from dotenv import load_dotenv
from openai import OpenAI

from pydantic import BaseModel, Field
from typing import Type

load_dotenv()

# ---------------------------
# OpenAI client
# ---------------------------
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

_DEFAULT_MODEL = os.getenv("OPENAI_MODEL")


def generate_fn(prompt: str, model: str | None = _DEFAULT_MODEL) -> str:
    """Plain text generation (kept for compatibility)."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def generate_structured_fn(
    prompt: str,
    output_model: Type[BaseModel],
    model: str | None = _DEFAULT_MODEL,
) -> BaseModel:
    """
    Native structured outputs with Pydantic.
    Returns a *parsed* instance of `output_model`.
    """
    # Modern path: native parse with Pydantic model
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise assistant. Follow the schema exactly; "
                    "no extra fields; do not invent items not supported by the prompt."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        # The SDK converts this Pydantic model to a strict JSON Schema and validates the reply.
        response_format=output_model,
    )
    msg = completion.choices[0].message
    if msg.parsed is not None:
        return msg.parsed  # already an instance of `output_model`
    # If the model refused or didn't parse, raise to let caller handle fallback.
    raise ValueError(msg.refusal or "Structured parse failed")
