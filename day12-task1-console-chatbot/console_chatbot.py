"""Task 1: a console chatbot built on the Anthropic SDK.

Follows the algorithm from the assessment:
    Step 1: get the input from the user
    Step 2: prepare the message object
    Step 3: invoke the LLM
    Step 4: print it in the console (question and answer)
    Step 5: quit

The conversation is kept in `messages`, so Claude remembers earlier turns.
Run:  python console_chatbot.py   (type "exit" or "quit" to stop)
"""

from pathlib import Path

import anthropic
from dotenv import load_dotenv

# ANTHROPIC_API_KEY comes from the .env file next to this script
load_dotenv(Path(__file__).resolve().parent / ".env")

client = anthropic.Anthropic()
MODEL = "claude-opus-5-5"
SYSTEM_PROMPT = "You are a friendly assistant. Keep answers short and clear, because they are shown in a terminal."
EXIT_WORDS = {"exit", "quit", "bye"}


def add_user_message(messages, text):
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages, content):
    # Keep every content block (thinking included), not only the text, so Claude's
    # reasoning from earlier turns carries over to the next request
    messages.append({"role": "assistant", "content": content})


def chat(messages):
    """Step 3: send the whole conversation to Claude and print the answer as it streams in."""
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=messages,
        output_config={"effort": "low"},  # low effort keeps a chat fast and cheap
        # If a safety classifier declines, the API retries on a recommended model
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        print()
        return stream.get_final_message()


def main():
    messages = []
    print(f"Chatting with {MODEL}. Type 'exit' to quit.\n")

    while True:
        # Step 1: get the input from the user
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            question = "exit"
        if not question:
            continue

        # Step 5: quit
        if question.lower() in EXIT_WORDS:
            print("Claude: Goodbye!")
            break

        # Step 2: prepare the message object
        add_user_message(messages, question)

        # Steps 3 and 4: invoke the LLM and print the answer
        print("Claude: ", end="", flush=True)
        try:
            response = chat(messages)
        except anthropic.AuthenticationError:
            print("\n[Error] Invalid or missing API key. Put ANTHROPIC_API_KEY in the .env file.")
            break
        except anthropic.RateLimitError:
            print("\n[Error] Rate limited. Wait a moment and try again.")
            messages.pop()
            continue
        except anthropic.APIConnectionError:
            print("\n[Error] Couldn't reach the Anthropic API. Check your network connection.")
            messages.pop()
            continue
        except anthropic.APIStatusError as e:
            print(f"\n[Error] API error {e.status_code}: {e.message}")
            messages.pop()
            continue

        if response.stop_reason == "refusal":
            # Drop the declined question so the conversation stays valid
            print("\n[Claude declined to answer this. Try rephrasing.]")
            messages.pop()
            continue

        add_assistant_message(messages, response.content)
        print()


if __name__ == "__main__":
    main()
