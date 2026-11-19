"""Task 2: IT Helpdesk Assistant, the Streamlit chat interface.

Start the MCP server first, then the app (from this folder, in two terminals):
    python mcp_server.py
    streamlit run streamlit.py
"""

import json

import anthropic
import streamlit as st

from app_implementation import MCP_SERVER_URL, MODEL, FAQKnowledgeBase, HelpdeskAssistant

SAMPLE_PROMPTS = [
    "How do I fix network connectivity issues?",
    "My computer is running slow. What should I do?",
    "How do I configure my email?",
    "How can I protect my laptop from malware?",
    "Search for the latest Windows 11 Wi-Fi troubleshooting steps.",
    "Find solutions for Outlook not syncing emails.",
    "Create a ticket for my laptop running slowly.",
    "Check the status of ticket IT-101.",
    "What is the current date and time?",
]
STEP_ICONS = {"MCP": "🎫", "client": "🕒", "server": "🌐"}


@st.cache_resource(show_spinner="Indexing the IT FAQ...")
def get_knowledge_base() -> FAQKnowledgeBase:
    """Embed the FAQ once and share it across sessions."""
    return FAQKnowledgeBase()


def get_assistant() -> HelpdeskAssistant:
    """One assistant (and conversation) per browser session."""
    if "assistant" not in st.session_state:
        st.session_state.assistant = HelpdeskAssistant(get_knowledge_base())
    return st.session_state.assistant


def render_details(turn: dict):
    """Tool calls and sources under an assistant answer."""
    if turn.get("notice"):
        st.warning(turn["notice"])
    if turn.get("steps"):
        with st.expander(f"Tools used ({len(turn['steps'])})"):
            for step in turn["steps"]:
                icon = "⚠️" if step.is_error else STEP_ICONS.get(step.kind, "🔧")
                st.markdown(f"{icon} **{step.name}** ({step.kind} tool)")
                st.json(step.input, expanded=False)
                try:
                    st.json(json.loads(step.output))  # ticket details from the MCP server
                except (ValueError, TypeError):
                    if step.output:
                        st.code(step.output, language=None)
    if turn.get("faq_sources") or turn.get("web_sources"):
        with st.expander("Sources"):
            for source in turn.get("faq_sources", []):
                st.markdown(f"📘 IT FAQ: **{source['title']}** (distance {source['distance']})")
            for source in turn.get("web_sources", []):
                st.markdown(f"🌐 [{source['title']}]({source['url']})")


def render_turn(turn: dict):
    with st.chat_message(turn["role"]):
        st.markdown(turn["text"])
        if turn["role"] == "assistant":
            render_details(turn)


def sidebar(assistant: HelpdeskAssistant) -> str | None:
    """Draw the sidebar. Returns a sample prompt if one was clicked."""
    with st.sidebar:
        st.header("🛠️ IT Helpdesk")
        st.caption(f"Model: {MODEL}")

        if assistant.mcp_tools:
            names = ", ".join(tool["name"] for tool in assistant.mcp_tools)
            st.success(f"MCP server connected. Tools: {names}")
        else:
            st.error(
                f"MCP server not reachable at {MCP_SERVER_URL}, so ticket tools are off. "
                "Start it with `python mcp_server.py`, then click Reconnect."
            )
        st.caption(f"RAG: {len(get_knowledge_base())} FAQ sections in the vector database")

        col1, col2 = st.columns(2)
        if col1.button("New chat", use_container_width=True):
            assistant.reset()
            st.session_state.chat = []
            st.rerun()
        if col2.button("Reconnect", use_container_width=True):
            st.session_state.pop("assistant", None)
            st.session_state.chat = []
            st.rerun()

        st.subheader("Try a sample prompt")
        for prompt in SAMPLE_PROMPTS:
            if st.button(prompt, use_container_width=True):
                return prompt
    return None


def main():
    st.set_page_config(page_title="IT Helpdesk Assistant", page_icon="🛠️")
    st.title("🛠️ IT Helpdesk Assistant")
    st.caption("Ask about IT problems, search for solutions online, or create and check support tickets.")

    try:
        assistant = get_assistant()
    except Exception as e:  # usually a missing or invalid VOYAGE_API_KEY
        st.error(f"Couldn't build the FAQ knowledge base: {e}\n\nCheck VOYAGE_API_KEY in the .env file.")
        st.stop()

    st.session_state.setdefault("chat", [])
    clicked = sidebar(assistant)

    for turn in st.session_state.chat:
        render_turn(turn)

    question = st.chat_input("Describe your IT problem...") or clicked
    if not question:
        return

    user_turn = {"role": "user", "text": question}
    render_turn(user_turn)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Working on it..."):
                reply = assistant.ask(question)
        except anthropic.AuthenticationError:
            st.error("Invalid or missing ANTHROPIC_API_KEY. Check the .env file.")
            return
        except anthropic.RateLimitError:
            st.error("Rate limited. Wait a moment and try again.")
            return
        except anthropic.APIStatusError as e:
            st.error(f"API error {e.status_code}: {e.message}")
            return
        except anthropic.APIConnectionError:
            st.error("Couldn't reach the Anthropic API. Check your network connection.")
            return
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            return

        assistant_turn = {
            "role": "assistant",
            "text": reply.text,
            "steps": reply.steps,
            "faq_sources": reply.faq_sources,
            "web_sources": reply.web_sources,
            "notice": reply.notice,
        }
        st.markdown(reply.text)
        render_details(assistant_turn)

    if not reply.refused:
        st.session_state.chat += [user_turn, assistant_turn]


if __name__ == "__main__":
    main()
