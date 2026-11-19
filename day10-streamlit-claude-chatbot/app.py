"""Day-10 homework: a Streamlit chat app for Claude.

The sidebar picks the model, effort, thinking and system prompt. Every time you send a
message, build_request() combines those settings with the conversation into one request.

Run from this folder:  streamlit run app.py
"""

import json
from pathlib import Path

import anthropic
import streamlit as st
from dotenv import load_dotenv

# ANTHROPIC_API_KEY lives in a .env file next to this script (copy .env.example to .env)
load_dotenv(Path(__file__).resolve().parent / ".env")

MAX_TOKENS = 64000  # streaming, so a large cap is safe; you only pay for tokens generated
EFFORT_LEVELS = ["low", "medium", "high", "xhigh", "max"]
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# The models differ in how thinking can be turned off, so each one carries its own rules:
#   thinking_off  - the `thinking` value that turns thinking off, or None if it can't be turned off
#   max_effort_off - the highest effort allowed while thinking is off
#   fallback      - whether the API can retry a refused request on another model
MODELS = {
    "Claude Opus 5.5": {
        "id": "claude-opus-5-5",
        "thinking_off": None,  # always thinks; lower the effort instead
        "default_effort": "medium",
        "fallback": True,
        "note": "Thinking is always on for this model. Use effort to control how much it thinks.",
    },
    "Claude Sonnet 5.5": {
        "id": "claude-sonnet-5-5",
        "thinking_off": {"type": "between_tools"},
        "max_effort_off": "high",
        "default_effort": "high",
        "fallback": True,
    },
    "Claude Haiku 5.5": {
        "id": "claude-haiku-5-5",
        "thinking_off": {"type": "disabled"},
        "max_effort_off": "high",
        "default_effort": "medium",
        "fallback": False,  # Haiku has no server-side fallback
    },
    "Claude Fable 5.2": {
        "id": "claude-fable-5-2",
        "thinking_off": None,  # always thinks
        "default_effort": "high",
        "fallback": True,
        "note": "Most capable model and the most expensive. Thinking is always on.",
    },
}


@st.cache_resource
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def sidebar_settings() -> dict:
    """Draw the sidebar and return the settings the user picked."""
    with st.sidebar:
        st.header("Settings")

        model_name = st.selectbox("Model", list(MODELS))
        model = MODELS[model_name]
        if "note" in model:
            st.caption(model["note"])

        can_disable = model["thinking_off"] is not None
        thinking = st.toggle(
            "Thinking",
            value=True,
            disabled=not can_disable,
            key=f"thinking_{model['id']}",
            help="When on, Claude reasons before answering. A summary appears above the answer.",
        )

        # With thinking off, the API rejects the two highest effort levels
        levels = EFFORT_LEVELS
        if not thinking:
            levels = EFFORT_LEVELS[: EFFORT_LEVELS.index(model["max_effort_off"]) + 1]
        # Remember the choice per model; lower it only when it isn't allowed any more
        effort_key = f"effort_{model['id']}"
        if effort_key not in st.session_state:
            st.session_state[effort_key] = model["default_effort"]
        if st.session_state[effort_key] not in levels:
            st.session_state[effort_key] = levels[-1]
        effort = st.select_slider(
            "Effort",
            options=levels,
            key=effort_key,
            help="How much work Claude puts in: thinking depth, length and token spend.",
        )

        system = st.text_area(
            "System instruction",
            value="You are a helpful assistant. Answer clearly and concisely.",
            height=150,
        )

        if st.button("New conversation", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    return {"model": model, "thinking": thinking, "effort": effort, "system": system.strip()}


def api_messages(history: list[dict], system: str) -> list[dict]:
    """The conversation in the shape the API expects.

    Assistant turns are sent back with all their blocks, thinking included. A thinking block
    is only valid while the system prompt stays the same as when it was produced, so if the
    user edits the system instruction mid-conversation, older thinking blocks are left out.
    """
    messages = []
    for turn in history:
        content = turn["content"]
        if turn["role"] == "assistant" and turn["system"] != system:
            content = [b for b in content if b.type not in ("thinking", "redacted_thinking")]
        messages.append({"role": turn["role"], "content": content})
    return messages


def build_request(settings: dict, history: list[dict]) -> dict:
    """Requirement 6: combine model, messages, system, effort and thinking into one request."""
    model = settings["model"]
    request = {
        "model": model["id"],
        "max_tokens": MAX_TOKENS,
        "messages": api_messages(history, settings["system"]),
        "output_config": {"effort": settings["effort"]},
    }
    if settings["system"]:
        request["system"] = settings["system"]

    if settings["thinking"]:
        # "summarized" returns a readable summary of the reasoning (the default shows nothing)
        request["thinking"] = {"type": "adaptive", "display": "summarized"}
    else:
        request["thinking"] = model["thinking_off"]

    if model["fallback"]:
        # If a safety classifier declines, the API retries on a recommended model
        request["betas"] = [FALLBACK_BETA]
        request["fallbacks"] = "default"
    return request


def request_preview(request: dict) -> dict:
    """The request without the full conversation, for the 'Request sent' expander."""
    preview = {k: v for k, v in request.items() if k != "messages"}
    latest = request["messages"][-1]["content"]
    preview["messages"] = f"{len(request['messages'])} message(s), latest: {latest!r}"
    return preview


def render_turn(turn: dict):
    with st.chat_message(turn["role"]):
        if turn.get("thinking"):
            with st.expander("Thinking"):
                st.markdown(turn["thinking"])
        st.markdown(turn["text"])
        if turn.get("meta"):
            st.caption(turn["meta"])


def stream_reply(request: dict) -> dict:
    """Send the request, stream the reply into the page and return the finished turn."""
    # Thinking always arrives before the answer, so each box is created when its first text
    # arrives and lands in the right order on the page
    thinking_box = answer_box = None
    thinking_text, answer_text = "", ""

    with get_client().beta.messages.stream(**request) as stream:
        for event in stream:
            if event.type != "content_block_delta":
                continue
            if event.delta.type == "thinking_delta" and event.delta.thinking:
                if thinking_box is None:
                    thinking_box = st.expander("Thinking", expanded=True).empty()
                thinking_text += event.delta.thinking
                thinking_box.markdown(thinking_text)
            elif event.delta.type == "text_delta":
                if answer_box is None:
                    answer_box = st.empty()
                answer_text += event.delta.text
                answer_box.markdown(answer_text + "▌")
        message = stream.get_final_message()

    if answer_box is not None:
        answer_box.markdown(answer_text)
    return {"message": message, "text": answer_text, "thinking": thinking_text}


def describe(message, settings: dict) -> str:
    thinking = "on" if settings["thinking"] else "off"
    meta = (
        f"{message.model} · effort {settings['effort']} · thinking {thinking} · "
        f"{message.usage.input_tokens} in / {message.usage.output_tokens} out tokens"
    )
    if message.model != settings["model"]["id"]:
        meta += " · answered by the fallback model"
    if message.stop_reason == "max_tokens":
        meta += " · cut off at max_tokens"
    return meta


def main():
    st.set_page_config(page_title="Claude Chat", page_icon="💬")
    st.title("💬 Claude Chat")

    st.session_state.setdefault("history", [])
    settings = sidebar_settings()

    for turn in st.session_state.history:
        render_turn(turn)

    question = st.chat_input("Ask Claude something")
    if not question:
        return

    history = st.session_state.history
    history.append({"role": "user", "content": question, "text": question})
    render_turn(history[-1])

    request = build_request(settings, history)
    with st.chat_message("assistant"):
        with st.expander("Request sent"):
            st.code(json.dumps(request_preview(request), indent=2), language="json")
        try:
            reply = stream_reply(request)
        except anthropic.AuthenticationError:
            history.pop()
            st.error("Invalid or missing API key. Set ANTHROPIC_API_KEY in the .env file next to app.py.")
            return
        except anthropic.PermissionDeniedError:
            history.pop()
            st.error("Your API key can't use this model. Try another model.")
            return
        except anthropic.BadRequestError as e:
            history.pop()
            st.error(f"The API rejected the request: {e.message}")
            return
        except anthropic.RateLimitError:
            history.pop()
            st.error("Rate limited. Wait a moment and send your message again.")
            return
        except anthropic.APIStatusError as e:
            history.pop()
            st.error(f"API error {e.status_code}: {e.message}")
            return
        except anthropic.APIConnectionError:
            history.pop()
            st.error("Couldn't reach the Anthropic API. Check your network connection.")
            return

        message = reply["message"]
        if message.stop_reason == "refusal":
            # Discard the partial answer and the question, so the conversation stays valid
            history.pop()
            category = message.stop_details.category if message.stop_details else None
            st.warning(f"Claude declined to answer this (category: {category}). Try rephrasing.")
            return

        meta = describe(message, settings)
        st.caption(meta)

    history.append(
        {
            "role": "assistant",
            "content": message.content,  # every block, sent back unchanged on the next turn
            "system": settings["system"],
            "text": reply["text"],
            "thinking": reply["thinking"],
            "meta": meta,
        }
    )


if __name__ == "__main__":
    main()
