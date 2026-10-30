"""Send a simple smoke-test prompt to the Luna and 5-mini models."""

import sys

from openai import APIError

from helper_functions import _get_client


MODELS = ("do-6-luna", "do-5-mini")
PROMPT = "Reply with exactly: OK"


def main() -> None:
    client = _get_client()
    failed = False

    for model in MODELS:
        print(f"Testing {model}...")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT}],
                max_tokens=16,
                temperature=0,
            )
        except APIError as exc:
            failed = True
            status = getattr(exc, "status_code", None)
            detail = f"HTTP {status}" if status is not None else type(exc).__name__
            print(f"  FAILED: {detail}")
            continue

        answer = response.choices[0].message.content
        if not answer:
            failed = True
            print("  FAILED: model returned an empty response")
            continue
        print(f"  OK: {answer.strip()}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
