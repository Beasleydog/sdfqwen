import os
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MAX_TOKENS = 131_072


def llm(
    model: str,
    messages: list[dict[str, str]],
    reasoning: Literal["low", "high", "max"] | bool | None = "max",
    include_reasoning: bool | None = False,
    max_tokens: int = MAX_TOKENS,
) -> tuple[str, float]:
    client = OpenAI(
        api_key=os.environ["PRIME_API_KEY"],
        base_url="https://api.pinference.ai/api/v1",
    )
    extra_body = {"usage": {"include": True}}
    if include_reasoning is not None:
        extra_body["include_reasoning"] = include_reasoning
    if isinstance(reasoning, bool):
        extra_body["reasoning"] = {"enabled": reasoning}

    request = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    if reasoning is not None and not isinstance(reasoning, bool):
        request["reasoning_effort"] = reasoning

    response = client.chat.completions.create(**request)
    cost = getattr(response.usage, "cost", 0.0) if response.usage else 0.0
    return response.choices[0].message.content or "", cost
