# AI Class Student Starter

Starter code for the class AI gateway. Your teacher will give you:

1. A gateway URL (ends with `/v1`)
2. Your personal key (`aiclass_...`)

Do **not** put the key in any file you commit. Do **not** share your key.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Credentials

```bash
export AI_CLASS_BASE_URL="https://ai.example.org/v1"   # ask your teacher
export AI_CLASS_API_KEY="aiclass_paste_your_key_here"
```

Optional: copy `.env.example` to `.env` and fill it in. `.env` is gitignored.

## Run

Ask one model:

```bash
python student_chat.py "Explain Newton's third law in two sentences."
```

Compare every model your key can see:

```bash
python compare_models.py "How many pineapples does it take to fill a bath tub?"
python compare_models.py --hosted-only "Skip the slow local model."
```

List models:

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["AI_CLASS_BASE_URL"],
    api_key=os.environ["AI_CLASS_API_KEY"],
)
for model in client.models.list().data:
    print(model.id)
```

Use only ids from that list (for example `local-small`, or hosted aliases your teacher enabled).

## Notes

- `max_tokens` controls how long the answer can be.
- Hosted models may accept `reasoning_effort` (`none`, `low`, `medium`, `high`, …). `local-small` does not.
- HTTP 429 means busy or rate-limited — wait and try again.
- If an error includes `request_id`, send that to your teacher. Never send your API key.

## Make your own copy

If this repository is a GitHub **template**, click **Use this template** → **Create a new repository**.

Or clone and work in a branch of your own fork.
