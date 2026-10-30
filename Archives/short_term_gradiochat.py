"""Launch a 5-mini chatbot with only the current chat's recent context."""

import gradio as gr

from helper_functions import get_chat_completion_with_usage


MODEL_ID = "do-5-mini"
MAX_TURNS = 15


def respond(message: str, history: list[dict[str, str]] | None):
    history = history or []
    if not message.strip():
        return history, "", "Tokens sent to model: —"

    recent_history = history[-(MAX_TURNS - 1) * 2 :]
    answer, input_tokens = get_chat_completion_with_usage(
        message,
        recent_history,
        model=MODEL_ID,
    )
    updated_history = recent_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    token_count = (
        "Tokens sent to model (system prompt + recent chat + your message): "
        f"**{input_tokens}**"
    )
    return updated_history, "", token_count


def main() -> None:
    with gr.Blocks(title="Short-Term 5-mini Chat") as app:
        gr.Markdown(
            "# Short-Term 5-mini Chat\n"
            "This chat uses 5-mini and keeps only the last 15 turns in the "
            "current session. It does not save chat history or create archives."
        )
        chatbot = gr.Chatbot(height=500)
        token_count = gr.Markdown("Tokens sent to model: —")
        with gr.Row():
            message = gr.Textbox(
                placeholder="Send a message",
                show_label=False,
                scale=6,
            )
            send = gr.Button("Send", variant="primary", scale=1)

        send.click(
            respond,
            inputs=[message, chatbot],
            outputs=[chatbot, message, token_count],
        )
        message.submit(
            respond,
            inputs=[message, chatbot],
            outputs=[chatbot, message, token_count],
        )
    app.launch(share=False)


if __name__ == "__main__":
    main()
