"""Ask every available class model the same question.

By default no reasoning_effort is sent (works for DeepSeek, GPT-4o mini, and
local-small). Pass --reasoning none|low|medium|high|... for hosted models that
support it. Pass --hosted-only to skip local-small.

    pip install openai

    export AI_CLASS_BASE_URL="https://ai.example.org/v1"
    export AI_CLASS_API_KEY="aiclass_paste_your_key_here"

    python compare_models.py "Explain Newton's third law in two sentences."
    python compare_models.py --hosted-only "Skip the slow local model."
    python compare_models.py --reasoning none "Same question with effort none."
"""

from __future__ import annotations

import os
import sys
import time

from openai import APIStatusError, OpenAI

MAX_TOKENS = 512
_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})


def main() -> None:
    base_url = os.environ.get("AI_CLASS_BASE_URL", "").strip()
    api_key = os.environ.get("AI_CLASS_API_KEY", "").strip()
    if not base_url or not api_key:
        sys.exit(
            "Set AI_CLASS_BASE_URL and AI_CLASS_API_KEY first. "
            "Ask your teacher for both. Do not paste the key into this file."
        )

    hosted_only, reasoning_effort, question_parts = _parse_args(sys.argv[1:])
    question = " ".join(question_parts).strip() or "Explain Newton's third law in two sentences."
    client = OpenAI(base_url=base_url, api_key=api_key)

    try:
        models = [item.id for item in client.models.list().data]
    except APIStatusError as exc:
        sys.exit(_format_error(exc))

    if hosted_only:
        models = [model for model in models if model != "local-small"]

    if not models:
        sys.exit("No models are available for your key.")

    print(f"Question: {question}")
    print(f"Models: {', '.join(models)}")
    if reasoning_effort is not None:
        print(f"reasoning_effort: {reasoning_effort} (hosted only)")
    print()

    for model in models:
        print("=" * 72)
        print(f"model={model}")
        print("-" * 72)
        started = time.perf_counter()
        try:
            kwargs: dict[str, object] = {
                "model": model,
                "messages": [{"role": "user", "content": question}],
                "max_tokens": MAX_TOKENS,
            }
            if reasoning_effort is not None and model != "local-small":
                kwargs["reasoning_effort"] = reasoning_effort
            response = client.chat.completions.create(**kwargs)
        except APIStatusError as exc:
            print(_format_error(exc))
            print()
            continue
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        content = response.choices[0].message.content or ""
        print(content if content else "(empty response)")
        print()
        print(f"({elapsed_ms} ms)")
        print()


def _parse_args(argv: list[str]) -> tuple[bool, str | None, list[str]]:
    hosted_only = False
    reasoning_effort: str | None = None
    question_parts: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--hosted-only":
            hosted_only = True
            index += 1
            continue
        if arg == "--reasoning":
            if index + 1 >= len(argv):
                sys.exit(
                    "Usage: compare_models.py [--hosted-only] [--reasoning EFFORT] [question...]"
                )
            effort = argv[index + 1]
            if effort not in _REASONING_EFFORTS:
                allowed = ", ".join(sorted(_REASONING_EFFORTS))
                sys.exit(f"reasoning_effort must be one of: {allowed}")
            reasoning_effort = effort
            index += 2
            continue
        question_parts.append(arg)
        index += 1
    return hosted_only, reasoning_effort, question_parts


def _format_error(exc: APIStatusError) -> str:
    if exc.status_code == 429:
        return "The model is busy. Wait a few seconds and run this again."
    detail = ""
    request_id = ""
    if isinstance(exc.body, dict):
        error = exc.body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            code = error.get("code")
            if isinstance(message, str) and message:
                detail = f" {message}"
            if isinstance(code, str) and code:
                detail = f" [{code}]{detail}"
            if error.get("request_id"):
                request_id = f" Request id: {error['request_id']}."
    return f"The gateway returned HTTP {exc.status_code}.{detail}{request_id}"


if __name__ == "__main__":
    main()
