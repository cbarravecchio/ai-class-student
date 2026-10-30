"""Ask the class model one question.

Your teacher gives you two values. Do not put them in this file, and do not
commit them.

    pip install openai

    export AI_CLASS_BASE_URL="https://ai.example.org/v1"
    export AI_CLASS_API_KEY="aiclass_paste_your_key_here"

    python student_chat.py
"""

import os
import sys

from openai import APIStatusError, OpenAI


def main() -> None:
    base_url = os.environ.get("AI_CLASS_BASE_URL", "").strip()
    api_key = os.environ.get("AI_CLASS_API_KEY", "").strip()
    if not base_url or not api_key:
        sys.exit(
            "Set AI_CLASS_BASE_URL and AI_CLASS_API_KEY first. "
            "Ask your teacher for both. Do not paste the key into this file."
        )

    question = " ".join(sys.argv[1:]).strip() or "Explain Newton's third law in two sentences."
    client = OpenAI(base_url=base_url, api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="local-small",
            messages=[{"role": "user", "content": question}],
            max_tokens=512,
        )
    except APIStatusError as exc:
        if exc.status_code == 429:
            sys.exit("The model is busy. Wait a few seconds and run this again.")
        request_id = ""
        if isinstance(exc.body, dict):
            error = exc.body.get("error")
            if isinstance(error, dict) and error.get("request_id"):
                request_id = f" Request id: {error['request_id']}."
        sys.exit(f"The gateway returned HTTP {exc.status_code}.{request_id}")

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
