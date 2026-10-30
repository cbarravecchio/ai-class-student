"""Launch a Gradio chatbot using the configured class AI model."""

import gradio as gr

from master_memory import (
    add_memory_claim,
    create_conversation,
    delete_conversation,
    delete_memory_claim,
    initialize_database,
    list_conversations,
    list_conversation_summaries,
    list_memory_claims,
    load_conversation,
    link_memory_claims,
    list_memory_conflicts,
    list_memory_relations,
    record_retrieval_feedback,
    resolve_memory_conflict,
    save_conversation_summary,
    search_memory,
    save_conversation,
    update_memory_claim,
)
from helper_functions import (
    extract_memory_candidates,
    get_chat_completion_with_token_usage,
    summarize_user_chats,
)


MODELS = {
    "Luna": "do-6-luna",
    "5-mini": "do-5-mini",
}
MODEL_CHOICES = (*MODELS, "Both")


def _past_chat_context(prompt, history):
    results, event_id, feedback_choices = search_memory(
        prompt,
        history=history,
        include_event_id=True,
        include_feedback_choices=True,
    )
    evidence = []
    for index, result in enumerate(results, start=1):
        citation = f"[M{index}]"
        source = (
            f"{result['title']}, {result['date']}, "
            f"message {result['source_message_index'] + 1}"
        )
        if result["kind"] in {"claim", "related_claim"}:
            relation = (
                f" Related via '{result['relation_type']}' "
                f"(hop {result['relation_depth']})."
                if result["kind"] == "related_claim"
                else ""
            )
            evidence.append(
                f"{citation} Confirmed memory ({result['category']}): "
                f"{result['content'][:1200]}{relation}\n"
                f"Source: {source}. Original user message: "
                f"{result['source_excerpt'][:1200]}"
            )
        else:
            evidence.append(
                f"{citation} Past chat ({result['role']}): "
                f"{result['content'][:1200]}\n"
                f"Source: {source}"
            )
    return (
        "\n\n".join(evidence),
        "\n\n".join(f"- {excerpt}" for excerpt in evidence),
        event_id,
        feedback_choices,
    )


def respond(prompt, history, conversation_id, model_choice, session_token_total=0):
    history = history or []
    if not prompt.strip():
        return (
            history,
            "",
            conversation_id,
            gr.Dropdown(choices=list_conversations()),
            gr.update(visible=False),
            "",
            "",
            None,
            "",
            None,
            gr.update(choices=[], value=None),
            _token_usage_text(None, None, session_token_total),
            session_token_total,
        )

    (
        past_chat_context,
        evidence_text,
        retrieval_event_id,
        feedback_choices,
    ) = _past_chat_context(prompt, history)
    if model_choice == "Both":
        luna_answer, luna_input_tokens, luna_output_tokens = (
            get_chat_completion_with_token_usage(
                prompt, history, past_chat_context, MODELS["Luna"]
            )
        )
        mini_answer, mini_input_tokens, mini_output_tokens = (
            get_chat_completion_with_token_usage(
                prompt, history, past_chat_context, MODELS["5-mini"]
            )
        )
        total_input_tokens = luna_input_tokens + mini_input_tokens
        total_output_tokens = luna_output_tokens + mini_output_tokens
        request_tokens = total_input_tokens + total_output_tokens
        updated_session_total = session_token_total + request_tokens
        token_usage_text = _token_usage_text(
            total_input_tokens, total_output_tokens, updated_session_total
        )
        pending = {
            "prompt": prompt,
            "history": history,
            "conversation_id": conversation_id,
            "evidence_text": evidence_text,
            "retrieval_event_id": retrieval_event_id,
            "feedback_choices": feedback_choices,
            "token_usage_text": token_usage_text,
            "session_token_total": updated_session_total,
        }
        return (
            history,
            "",
            conversation_id,
            gr.Dropdown(choices=list_conversations(), value=conversation_id),
            gr.update(visible=True),
            luna_answer,
            mini_answer,
            pending,
            evidence_text,
            retrieval_event_id,
            gr.update(choices=feedback_choices, value=None),
            token_usage_text,
            updated_session_total,
        )

    if model_choice not in MODELS:
        raise ValueError(f"Unsupported model selection: {model_choice!r}")
    answer, input_tokens, output_tokens = get_chat_completion_with_token_usage(
        prompt, history, past_chat_context, MODELS[model_choice]
    )
    updated_session_total = session_token_total + input_tokens + output_tokens
    return _save_selected_answer(
        prompt,
        answer,
        history,
        conversation_id,
        evidence_text,
        retrieval_event_id,
        feedback_choices,
        _token_usage_text(input_tokens, output_tokens, updated_session_total),
        updated_session_total,
    )


def _token_usage_text(input_tokens, output_tokens, session_total):
    latest_input = "—" if input_tokens is None else f"{input_tokens:,}"
    latest_output = "—" if output_tokens is None else f"{output_tokens:,}"
    return (
        f"**Tokens from latest input:** {latest_input}  \n"
        f"**Tokens from latest output:** {latest_output}  \n"
        f"**Total tokens used this session:** {session_total:,}"
    )


def _save_selected_answer(
    prompt,
    answer,
    history,
    conversation_id,
    evidence_text="",
    retrieval_event_id=None,
    feedback_choices=None,
    token_usage_text="",
    session_token_total=0,
):
    conversation_id = conversation_id or create_conversation()
    messages = list(history)
    messages.extend(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    )
    save_conversation(conversation_id, messages)
    return (
        messages,
        "",
        conversation_id,
        gr.Dropdown(choices=list_conversations(), value=conversation_id),
        gr.update(visible=False),
        "",
        "",
        None,
        evidence_text,
        retrieval_event_id,
        gr.update(choices=feedback_choices or [], value=None),
        token_usage_text,
        session_token_total,
    )


def choose_preferred_response(selected_model, luna_answer, mini_answer, pending):
    if pending is None:
        raise ValueError("No pending model responses to choose from.")
    if selected_model == "Luna":
        answer = luna_answer
    elif selected_model == "5-mini":
        answer = mini_answer
    else:
        raise ValueError(f"Unsupported preferred response: {selected_model!r}")
    return _save_selected_answer(
        pending["prompt"],
        answer,
        pending["history"],
        pending["conversation_id"],
        pending.get("evidence_text", ""),
        pending.get("retrieval_event_id"),
        pending.get("feedback_choices"),
        pending.get("token_usage_text", ""),
        pending.get("session_token_total", 0),
    )


def start_new_chat():
    return (
        [],
        None,
        gr.Dropdown(choices=list_conversations(), value=None),
        gr.update(visible=False),
        "",
        "",
        None,
        "",
        None,
        gr.update(choices=list_conversations(), value=None),
        gr.update(choices=[], value=None),
    )


def open_conversation(conversation_id):
    if not conversation_id:
        return (
            [],
            None,
            gr.update(visible=False),
            "",
            "",
            None,
            "",
            None,
            gr.update(choices=list_conversations(), value=None),
            gr.update(choices=[], value=None),
        )
    return (
        load_conversation(conversation_id),
        conversation_id,
        gr.update(visible=False),
        "",
        "",
        None,
        "",
        None,
        gr.update(
            choices=list_conversations(),
            value=conversation_id,
        ),
        gr.update(choices=[], value=None),
    )


def summarize_past_chats():
    saved_summaries = {
        summary["conversation_id"]: summary
        for summary in list_conversation_summaries()
    }
    for label, conversation_id in list_conversations():
        existing = saved_summaries.get(conversation_id)
        if existing and not existing["stale"]:
            continue
        messages = load_conversation(conversation_id)
        user_messages = [
            (label, message["content"].strip())
            for message in messages
            if message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and message["content"].strip()
        ]
        if user_messages:
            save_conversation_summary(
                conversation_id, summarize_user_chats(user_messages)
            )
    saved_summaries = list_conversation_summaries()
    if not saved_summaries:
        return "No saved conversations to summarize yet.", ""
    profile = summarize_user_chats(
        [(item["title"], item["summary"]) for item in saved_summaries]
    )
    details = "\n\n".join(
        f"{'[STALE] ' if item['stale'] else ''}{item['title']}\n"
        f"{item['summary']}\nUpdated: {item['updated_at']}"
        for item in saved_summaries
    )
    return profile, details


def memory_claim_choices():
    return [
        (
            f"[{claim['status']}] {claim['content'][:72]} - {claim['source_title']}",
            claim["id"],
        )
        for claim in list_memory_claims()
    ]


def open_memory_conflicts():
    return [
        (
            f"{conflict['claim_a'][:50]} <> {conflict['claim_b'][:50]}",
            conflict["id"],
        )
        for conflict in list_memory_conflicts()
    ]


def describe_memory_conflict(conflict_id):
    conflict = next(
        (item for item in list_memory_conflicts() if item["id"] == conflict_id),
        None,
    )
    if conflict is None:
        return "Select an open conflict to inspect it."
    return (
        f"Potential conflict: {conflict['reason']}\n"
        f"Claim #{conflict['claim_a_id']} [{conflict['claim_a_status']}]: "
        f"{conflict['claim_a']}\n"
        f"Claim #{conflict['claim_b_id']} [{conflict['claim_b_status']}]: "
        f"{conflict['claim_b']}"
    )


def resolve_conflict_from_ui(conflict_id, resolution):
    if conflict_id is None:
        raise ValueError("Select an open memory conflict.")
    if resolution not in {"resolved", "dismissed"}:
        raise ValueError("Choose resolved or dismissed.")
    resolve_memory_conflict(conflict_id, resolution)
    return (
        f"Conflict marked {resolution}.",
        gr.update(choices=open_memory_conflicts(), value=None),
        "Select an open conflict to inspect it.",
    )


def confirmed_memory_choices():
    return [
        (f"#{claim['id']} {claim['content'][:90]}", claim["id"])
        for claim in list_memory_claims()
        if claim["status"] == "confirmed"
    ]


def create_memory_relation(source_id, target_id, relation_type):
    if source_id is None or target_id is None:
        raise ValueError("Select two confirmed memories to connect.")
    link_memory_claims(source_id, target_id, relation_type)
    relations = list_memory_relations()
    display = "\n".join(
        f"#{source_id}: {source} --{kind}--> #{target_id}: {target}"
        for source_id, source, kind, target_id, target in relations
    )
    return "Memory relation saved.", display


def source_message_choices(conversation_id):
    if not conversation_id:
        return gr.update(choices=[], value=None)
    messages = load_conversation(conversation_id)
    choices = [
        (f"Message {index + 1}: {message['content'][:70]}", index)
        for index, message in enumerate(messages)
        if message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    ]
    return gr.update(
        choices=choices,
        value=choices[0][1] if choices else None,
    )


def load_memory_claim(claim_id):
    claim = next(
        (item for item in list_memory_claims() if item["id"] == claim_id),
        None,
    )
    if claim is None:
        return "", "", "candidate", "Select a saved memory to inspect it."
    source = (
        f"Source: {claim['source_title']}, message "
        f"{claim['source_message_index'] + 1} ({claim['valid_from']}).\n"
        f"Original message: {claim['source_excerpt']}"
    )
    return claim["content"], claim["category"], claim["status"], source


def add_memory_from_ui(
    conversation_id, message_index, content, category, supersedes_id
):
    if not conversation_id or message_index is None:
        raise ValueError("Select a conversation and source user message.")
    claim_id = add_memory_claim(
        content,
        category,
        conversation_id,
        message_index,
        supersedes_id=supersedes_id,
    )
    return (
        f"Added memory candidate #{claim_id}. Confirm it to make it searchable.",
        gr.update(choices=memory_claim_choices(), value=claim_id),
        gr.update(choices=memory_claim_choices(), value=None),
        "",
        "",
        gr.update(choices=open_memory_conflicts()),
    )


def suggest_memories(conversation_id):
    if not conversation_id:
        raise ValueError("Select a saved conversation first.")
    messages = load_conversation(conversation_id)
    user_messages = [
        (index, message["content"].strip())
        for index, message in enumerate(messages)
        if message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    ]
    if not user_messages:
        return "This conversation has no user messages to review.", gr.update(
            choices=memory_claim_choices()
        ), gr.update(choices=memory_claim_choices()), gr.update(
            choices=open_memory_conflicts()
        )
    candidates = extract_memory_candidates(user_messages)
    created = 0
    for candidate in candidates:
        add_memory_claim(
            candidate["content"],
            candidate["category"],
            conversation_id,
            candidate["source_id"],
            confidence=0.5,
        )
        created += 1
    return f"Added {created} unconfirmed memory suggestion(s). Review and confirm them below.", gr.update(
        choices=memory_claim_choices()
    ), gr.update(choices=memory_claim_choices()), gr.update(
        choices=open_memory_conflicts()
    )


def save_memory(claim_id, content, category, status):
    if claim_id is None:
        raise ValueError("Select a memory suggestion first.")
    update_memory_claim(claim_id, content, category, status)
    return (
        f"Saved memory #{claim_id} as {status}.",
        gr.update(choices=memory_claim_choices(), value=claim_id),
        gr.update(choices=memory_claim_choices()),
        gr.update(choices=confirmed_memory_choices()),
        gr.update(choices=confirmed_memory_choices()),
        gr.update(choices=open_memory_conflicts()),
        "\n".join(
            f"#{source_id}: {source} --{kind}--> #{target_id}: {target}"
            for source_id, source, kind, target_id, target
            in list_memory_relations()
        ),
    )


def remove_memory(claim_id):
    if claim_id is None:
        raise ValueError("Select a memory to delete first.")
    delete_memory_claim(claim_id)
    return (
        "Memory deleted.",
        gr.update(choices=memory_claim_choices(), value=None),
        gr.update(choices=memory_claim_choices(), value=None),
        "",
        "",
        "candidate",
        "",
        gr.update(choices=open_memory_conflicts(), value=None),
        "",
        gr.update(choices=confirmed_memory_choices(), value=None),
        gr.update(choices=confirmed_memory_choices(), value=None),
        "\n".join(
            f"#{source_id}: {source} --{kind}--> #{target_id}: {target}"
            for source_id, source, kind, target_id, target
            in list_memory_relations()
        ),
    )


def remove_conversation(conversation_id):
    if not conversation_id:
        raise ValueError("Select a saved conversation first.")
    delete_conversation(conversation_id)
    return (
        [],
        None,
        gr.update(choices=list_conversations(), value=None),
        gr.update(visible=False),
        "",
        "",
        None,
        "",
        None,
        gr.update(choices=[], value=None),
        None,
        "Deleted the conversation, its search entries, and its source-linked memories.",
        "",
        "",
        gr.update(choices=list_conversations(), value=None),
        gr.update(choices=[], value=None),
        gr.update(choices=memory_claim_choices(), value=None),
        gr.update(choices=memory_claim_choices(), value=None),
        "",
        "",
        "candidate",
        "",
        gr.update(choices=open_memory_conflicts(), value=None),
        "",
        gr.update(choices=confirmed_memory_choices(), value=None),
        gr.update(choices=confirmed_memory_choices(), value=None),
        "",
        "",
    )


def save_retrieval_feedback(event_id, record_key, helpful):
    if event_id is None or record_key is None:
        raise ValueError("Select a retrieved evidence item to rate.")
    record_retrieval_feedback(event_id, record_key, helpful)
    return "Thanks, your retrieval feedback was saved locally."


def submit_retrieval_feedback(event_id, record_key, rating):
    if rating not in {"Helpful", "Not helpful"}:
        raise ValueError("Choose whether the selected evidence was helpful.")
    return save_retrieval_feedback(
        event_id, record_key, rating == "Helpful"
    )


def main() -> None:
    initialize_database()
    with gr.Blocks(title="Class AI Chatbot") as app:
        gr.Markdown(
            f"# Class AI Chatbot\nChoose Luna, 5-mini, or compare both. "
            "Your memory and chat history are stored locally in SQLite. "
            "Relevant messages from past chats are searched automatically. "
            "Use the summary button to review what you've shared across chats."
        )
        conversation_id = gr.State()
        pending_response = gr.State()
        retrieval_event = gr.State()
        luna_selection = gr.State("Luna")
        mini_selection = gr.State("5-mini")
        with gr.Row():
            with gr.Column(scale=3):
                model_choice = gr.Dropdown(
                    label="Responding model",
                    choices=MODEL_CHOICES,
                    value="5-mini",
                )
                chatbot = gr.Chatbot(height=500)
                token_count = gr.Markdown(
                    _token_usage_text(None, None, 0)
                )
                with gr.Row():
                    prompt = gr.Textbox(
                        placeholder="Send a message",
                        show_label=False,
                        scale=6,
                    )
                    send = gr.Button("Send", variant="primary", scale=1)
                with gr.Row():
                    new_chat = gr.Button("New chat")
                    past_chats = gr.Dropdown(
                        label="Past chats",
                        choices=list_conversations(),
                        value=None,
                    )
                with gr.Group(visible=False) as comparison:
                    gr.Markdown("### Choose your preferred response")
                    with gr.Row():
                        with gr.Column():
                            luna_answer = gr.Textbox(
                                label="Luna response",
                                lines=8,
                                interactive=False,
                            )
                            choose_luna = gr.Button("Use Luna response")
                        with gr.Column():
                            mini_answer = gr.Textbox(
                                label="5-mini response",
                                lines=8,
                                interactive=False,
                            )
                            choose_mini = gr.Button("Use 5-mini response")
            with gr.Column(scale=1):
                gr.Markdown(
                    "## Your past-chat summary\nGenerate an overview from your "
                    "saved conversations."
                )
                summary = gr.Textbox(
                    label="What I know about you",
                    lines=10,
                    placeholder="Click the button to summarize your saved chats.",
                    interactive=False,
                )
                summarize = gr.Button("Summarize my past chats")
                conversation_summaries = gr.Textbox(
                    label="Per-conversation summaries",
                    lines=12,
                    interactive=False,
                )
                gr.Markdown("## Memory review")
                memory_source = gr.Dropdown(
                    label="Conversation to extract suggestions from",
                    choices=list_conversations(),
                    value=None,
                )
                suggest_memories_button = gr.Button(
                    "Suggest memories from selected chat"
                )
                memory_status = gr.Markdown()
                memory_source_message = gr.Dropdown(
                    label="Source user message for a new memory",
                    choices=[],
                    value=None,
                )
                new_memory_content = gr.Textbox(
                    label="New memory claim", lines=2
                )
                new_memory_category = gr.Textbox(
                    label="New memory category"
                )
                memory_supersedes = gr.Dropdown(
                    label="Supersedes memory (optional)",
                    choices=memory_claim_choices(),
                    value=None,
                )
                add_memory_button = gr.Button("Add memory candidate")
                memory_claim = gr.Dropdown(
                    label="Saved memory claims",
                    choices=memory_claim_choices(),
                    value=None,
                )
                memory_content = gr.Textbox(label="Memory", lines=3)
                memory_category = gr.Textbox(label="Category")
                memory_state = gr.Dropdown(
                    label="Status",
                    choices=["candidate", "confirmed", "disputed", "superseded"],
                    value="candidate",
                )
                memory_source_details = gr.Textbox(
                    label="Source evidence",
                    lines=4,
                    interactive=False,
                )
                memory_conflict = gr.Dropdown(
                    label="Potential claim conflicts",
                    choices=open_memory_conflicts(),
                    value=None,
                )
                memory_conflict_details = gr.Textbox(
                    label="Conflict details",
                    lines=4,
                    interactive=False,
                )
                memory_conflict_resolution = gr.Dropdown(
                    label="Resolution",
                    choices=["resolved", "dismissed"],
                    value="resolved",
                )
                resolve_conflict_button = gr.Button("Resolve / dismiss conflict")
                memory_relations_display = gr.Textbox(
                    label="Memory relationships",
                    lines=4,
                    value="\n".join(
                        f"#{source_id}: {source} --{kind}--> "
                        f"#{target_id}: {target}"
                        for source_id, source, kind, target_id, target
                        in list_memory_relations()
                    ),
                    interactive=False,
                )
                relation_source = gr.Dropdown(
                    label="Related memory source",
                    choices=confirmed_memory_choices(),
                    value=None,
                )
                relation_target = gr.Dropdown(
                    label="Related memory target",
                    choices=confirmed_memory_choices(),
                    value=None,
                )
                relation_type = gr.Textbox(
                    label="Relationship (e.g. 'belongs to project')"
                )
                add_relation_button = gr.Button("Connect memories")
                with gr.Row():
                    save_memory_button = gr.Button("Save / update memory")
                    delete_memory_button = gr.Button("Delete memory", variant="stop")
                gr.Markdown(
                    "Deleting a conversation also deletes its source-linked memories."
                )
                delete_chat_button = gr.Button(
                    "Delete selected conversation and memories", variant="stop"
                )
                delete_chat_status = gr.Markdown()
                evidence = gr.Textbox(
                    label="Memory evidence used for the last answer",
                    lines=7,
                    interactive=False,
                )
                with gr.Row():
                    evidence_feedback_item = gr.Dropdown(
                        label="Rate an evidence item",
                        choices=[],
                        value=None,
                    )
                    evidence_feedback_rating = gr.Dropdown(
                        label="Was it helpful?",
                        choices=["Helpful", "Not helpful"],
                        value=None,
                    )
                    rate_evidence_button = gr.Button("Save feedback")
                retrieval_feedback_status = gr.Markdown()

        session_token_total = gr.State(0)
        send_inputs = [
            prompt,
            chatbot,
            conversation_id,
            model_choice,
            session_token_total,
        ]
        send_outputs = [
            chatbot,
            prompt,
            conversation_id,
            past_chats,
            comparison,
            luna_answer,
            mini_answer,
            pending_response,
            evidence,
            retrieval_event,
            evidence_feedback_item,
            token_count,
            session_token_total,
        ]
        send.click(respond, inputs=send_inputs, outputs=send_outputs)
        prompt.submit(respond, inputs=send_inputs, outputs=send_outputs)
        new_chat.click(
            start_new_chat,
            outputs=[
                chatbot,
                conversation_id,
                past_chats,
                comparison,
                luna_answer,
                mini_answer,
                pending_response,
                evidence,
                retrieval_event,
                memory_source,
                evidence_feedback_item,
            ],
        )
        past_chats.change(
            open_conversation,
            inputs=past_chats,
            outputs=[
                chatbot,
                conversation_id,
                comparison,
                luna_answer,
                mini_answer,
                pending_response,
                evidence,
                retrieval_event,
                memory_source,
                evidence_feedback_item,
            ],
        )
        choose_mini.click(
            choose_preferred_response,
            inputs=[mini_selection, luna_answer, mini_answer, pending_response],
            outputs=send_outputs,
        )
        choose_luna.click(
            choose_preferred_response,
            inputs=[luna_selection, luna_answer, mini_answer, pending_response],
            outputs=send_outputs,
        )
        summarize.click(
            summarize_past_chats,
            outputs=[summary, conversation_summaries],
        )
        memory_source.change(
            source_message_choices,
            inputs=memory_source,
            outputs=memory_source_message,
        )
        add_memory_button.click(
            add_memory_from_ui,
            inputs=[
                memory_source,
                memory_source_message,
                new_memory_content,
                new_memory_category,
                memory_supersedes,
            ],
            outputs=[
                memory_status,
                memory_claim,
                memory_supersedes,
                new_memory_content,
                new_memory_category,
                memory_conflict,
            ],
        )
        memory_claim.change(
            load_memory_claim,
            inputs=memory_claim,
            outputs=[
                memory_content,
                memory_category,
                memory_state,
                memory_source_details,
            ],
        )
        memory_claim.change(
            lambda _: gr.update(choices=confirmed_memory_choices()),
            inputs=memory_claim,
            outputs=relation_source,
        )
        memory_claim.change(
            lambda _: gr.update(choices=confirmed_memory_choices()),
            inputs=memory_claim,
            outputs=relation_target,
        )
        memory_claim.change(
            lambda _: gr.update(choices=open_memory_conflicts()),
            inputs=memory_claim,
            outputs=memory_conflict,
        )
        memory_conflict.change(
            describe_memory_conflict,
            inputs=memory_conflict,
            outputs=memory_conflict_details,
        )
        resolve_conflict_button.click(
            resolve_conflict_from_ui,
            inputs=[memory_conflict, memory_conflict_resolution],
            outputs=[
                memory_status,
                memory_conflict,
                memory_conflict_details,
            ],
        )
        add_relation_button.click(
            create_memory_relation,
            inputs=[relation_source, relation_target, relation_type],
            outputs=[memory_status, memory_relations_display],
        )
        suggest_memories_button.click(
            suggest_memories,
            inputs=memory_source,
            outputs=[
                memory_status,
                memory_claim,
                memory_supersedes,
                memory_conflict,
            ],
        )
        save_memory_button.click(
            save_memory,
            inputs=[memory_claim, memory_content, memory_category, memory_state],
            outputs=[
                memory_status,
                memory_claim,
                memory_supersedes,
                relation_source,
                relation_target,
                memory_conflict,
                memory_relations_display,
            ],
        )
        delete_memory_button.click(
            remove_memory,
            inputs=memory_claim,
            outputs=[
                memory_status,
                memory_claim,
                memory_supersedes,
                memory_content,
                memory_category,
                memory_state,
                memory_source_details,
                memory_conflict,
                memory_conflict_details,
                relation_source,
                relation_target,
                memory_relations_display,
            ],
        )
        delete_chat_button.click(
            remove_conversation,
            inputs=past_chats,
            outputs=[
                chatbot,
                conversation_id,
                past_chats,
                comparison,
                luna_answer,
                mini_answer,
                pending_response,
                evidence,
                retrieval_event,
                evidence_feedback_item,
                evidence_feedback_rating,
                delete_chat_status,
                summary,
                conversation_summaries,
                memory_source,
                memory_source_message,
                memory_claim,
                memory_supersedes,
                memory_content,
                memory_category,
                memory_state,
                memory_source_details,
                memory_conflict,
                memory_conflict_details,
                relation_source,
                relation_target,
                memory_relations_display,
                relation_type,
            ],
        )
        rate_evidence_button.click(
            submit_retrieval_feedback,
            inputs=[
                retrieval_event,
                evidence_feedback_item,
                evidence_feedback_rating,
            ],
            outputs=retrieval_feedback_status,
        )
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()