# Task 1: Console Chatbot

A chatbot that runs in the terminal. You type a question, Claude answers, and the conversation continues until you type `exit`. Claude remembers earlier messages in the conversation.

## Run it

```bash
cd day12-task1-console-chatbot

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then put your ANTHROPIC_API_KEY in .env
python console_chatbot.py
```

Example session:

```text
Chatting with claude-opus-5-5. Type 'exit' to quit.

You: My name is Ana.
Claude: Nice to meet you, Ana! How can I help you today?

You: What is my name?
Claude: Your name is Ana.

You: exit
Claude: Goodbye!
```

## How it follows the algorithm

| Step | In [console_chatbot.py](console_chatbot.py) |
|---|---|
| 1. Get the input from the user | `input("You: ")` |
| 2. Prepare the message object | `add_user_message(messages, question)` adds `{"role": "user", "content": ...}` to the conversation |
| 3. Invoke the LLM | `chat(messages)` sends the whole conversation to Claude |
| 4. Print the question and answer | The question stays on screen as typed, and the answer streams in after `Claude:` |
| 5. Quit | Typing `exit`, `quit` or `bye` (or pressing Ctrl+C) ends the loop |

**Memory:** the API is stateless, so every request sends the full `messages` list. After each answer, `add_assistant_message` appends Claude's reply, so the next question has the whole conversation as context.
