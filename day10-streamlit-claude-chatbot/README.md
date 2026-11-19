# Claude Chat (Day-10 homework)

A Streamlit chat app built on the Anthropic Python SDK. The sidebar sets the model, effort, thinking and system instruction. Every message you send combines those settings with the conversation into one request (`build_request()` in [app.py](app.py)).

## Run it

```bash
cd day10-streamlit-claude-chatbot

# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install the dependencies
pip install -r requirements.txt

# 3. Add your API key: copy the example file, then put your key in .env
cp .env.example .env

# 4. Run the app
streamlit run app.py
```

Streamlit opens the app at http://localhost:8501. The `.env` file is ignored by git, so your key stays on your machine.

## How each requirement is met

| Requirement | Where |
|---|---|
| Chat UI | `st.chat_input` plus `st.chat_message`. Replies stream in token by token. |
| Model selection | Sidebar select box: Opus 5.5, Sonnet 5.5, Haiku 5.5 or Fable 5.2 |
| Effort | Sidebar slider, sent as `output_config={"effort": ...}` |
| Thinking | Sidebar toggle. On sends `{"type": "adaptive", "display": "summarized"}`, and the summary appears in a "Thinking" expander above the answer. |
| System instruction | Sidebar text area, sent as `system` |
| Dynamic request | `build_request()` merges all of the above with the conversation history. The "Request sent" expander shows the request for each message. |

## Model differences the app handles

Each model has its own rules for thinking and effort, so the sidebar adjusts itself:

| Model | Turning thinking off | Effort with thinking off |
|---|---|---|
| Opus 5.5 | Not possible (the toggle is locked on). Lower the effort instead. | — |
| Sonnet 5.5 | `{"type": "between_tools"}` | low to high only |
| Haiku 5.5 | `{"type": "disabled"}` | low to high only |
| Fable 5.2 | Not possible (the toggle is locked on) | — |

Other details:

- **Memory:** the API is stateless, so the whole conversation is sent on every turn. Assistant turns go back with all their content blocks, thinking included.
- **Changing the system instruction mid-chat:** a thinking block is only valid while the system prompt is unchanged, so the app drops thinking blocks from turns produced under an older system instruction.
- **Refusals:** on Opus, Sonnet and Fable, `fallbacks="default"` (beta `server-side-fallback-2026-07-01`) lets the API retry a declined request on another model. If the reply is still a refusal, the app shows a warning and removes that exchange from the history.
- **Errors:** bad API key, no model access, rate limits and network problems each show a clear message instead of a stack trace.
