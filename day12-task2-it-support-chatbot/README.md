# Task 2: IT Helpdesk Assistant

An AI-powered IT helpdesk chatbot. Employees ask IT questions in a Streamlit chat. Claude answers from the company IT FAQ (RAG), searches the web for anything the FAQ doesn't cover, tells the current date and time, and creates or checks support tickets through an MCP server.

## Files

| File | Purpose |
|---|---|
| [streamlit.py](streamlit.py) | Chat interface: conversation, sample prompts, tools used, sources |
| [app_implementation.py](app_implementation.py) | RAG, tools, MCP client and the Claude tool-use loop |
| [mcp_server.py](mcp_server.py) | **IT Helpdesk MCP Server** with `create_support_ticket` and `check_ticket_status` |
| [it_sector.txt](it_sector.txt) | IT support FAQ used for RAG |
| [mcp_tool_data.json](mcp_tool_data.json) | Starting tickets (IT-101 to IT-103), loaded into the server's in-memory dictionary |
| [requirements.txt](requirements.txt) | Python dependencies |

## Run it

You need an Anthropic API key and a [Voyage AI](https://www.voyageai.com/) API key.

> **Voyage free tier:** until you add a payment method in the Voyage dashboard, the key is limited to 3 requests per minute. The free embedding tokens still apply after you add one. The app makes one Voyage request at startup and one per question. If a request is rate-limited, the answer still comes back without the FAQ, and the app shows a warning.

```bash
cd day12-task2-it-support-chatbot

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then put ANTHROPIC_API_KEY and VOYAGE_API_KEY in .env
```

Start the two parts in separate terminals, with the venv active in both:

```bash
# Terminal 1: the MCP server (http://localhost:3000/mcp)
python mcp_server.py

# Terminal 2: the chat app (http://localhost:8501)
streamlit run streamlit.py
```

Use `streamlit run streamlit.py`, not `python -m streamlit run streamlit.py`. Because this folder contains a file named `streamlit.py`, `python -m streamlit` would load that file instead of the Streamlit package.

## How it works

```text
Employee question
      │
      ▼
streamlit.py ──► app_implementation.py
                   1. RAG: embed the question (Voyage AI), find the closest FAQ sections in Chroma
                   2. Send the question + FAQ sections + conversation to Claude, with 4 tools
                   3. Tool loop until Claude has a final answer:
                        web_search            → run by Anthropic (server tool)
                        get_current_datetime  → run in app_implementation.py (client tool)
                        create_support_ticket → sent to mcp_server.py (MCP tool)
                        check_ticket_status   → sent to mcp_server.py (MCP tool)
```

| Requirement | Implementation |
|---|---|
| **RAG** | `FAQKnowledgeBase` splits `it_sector.txt` into one chunk per `##` section, embeds the chunks with Voyage AI (`voyage-3-large`) and stores them in a Chroma vector database. Each question retrieves the 2 closest sections, which go to Claude with the question. |
| **MCP** | `mcp_server.py` is a Python MCP server named "IT Helpdesk MCP Server". Its tickets live in an in-memory dictionary. The app discovers the server's tools at startup and calls them over Streamable HTTP. |
| **Tool calling** | `web_search` (server tool) and `get_current_datetime` (client tool) are offered to Claude alongside the MCP tools. Claude decides which to call, and the loop runs them. |
| **Chatbot interface** | Streamlit chat with conversation memory. Each answer has a "Tools used" panel (including ticket details) and a "Sources" panel (FAQ sections and web pages). The sidebar shows the MCP connection status and the sample prompts. |

## Sample prompts

These are in the sidebar, so you can click them:

| Category | Prompt | What happens |
|---|---|---|
| RAG | How do I fix network connectivity issues? | Answers from the IT FAQ |
| RAG | My computer is running slow. What should I do? | Answers from the IT FAQ |
| RAG | How do I configure my email? | Answers from the IT FAQ |
| RAG | How can I protect my laptop from malware? | Answers from the IT FAQ |
| Web search | Search for the latest Windows 11 Wi-Fi troubleshooting steps. | Calls `web_search` |
| Web search | Find solutions for Outlook not syncing emails. | Calls `web_search` |
| MCP | Create a ticket for my laptop running slowly. | Asks for your name, then calls `create_support_ticket` |
| MCP | Check the status of ticket IT-101. | Calls `check_ticket_status` |
| Tool calling | What is the current date and time? | Calls `get_current_datetime` |

New tickets are kept only while `mcp_server.py` is running. Restarting it resets the tickets to IT-101 to IT-103.
