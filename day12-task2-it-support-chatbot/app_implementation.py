"""Back end of the IT Helpdesk Assistant: RAG, tools, the MCP client and the Claude tool loop.

The pieces:
  - FAQKnowledgeBase  RAG. Splits it_sector.txt into one chunk per FAQ section, embeds them
                      with Voyage AI and stores them in a Chroma vector database.
  - Tools             web_search (server tool, run by Anthropic) and get_current_datetime
                      (client tool, run here).
  - MCP client        discovers and calls the IT Helpdesk MCP Server's ticket tools.
  - HelpdeskAssistant keeps the conversation and runs the Claude tool-use loop.

streamlit.py builds the UI on top of this file.
"""

import asyncio
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import anthropic
import chromadb
import voyageai
from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # ANTHROPIC_API_KEY and VOYAGE_API_KEY

MODEL = "claude-opus-5-5"
EMBEDDING_MODEL = "voyage-3-large"
FAQ_FILE = BASE_DIR / "it_sector.txt"
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:3000/mcp")
FAQ_RESULTS = 2  # FAQ sections retrieved per question
MAX_TOOL_ROUNDS = 10  # safety limit for the tool-use loop
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """You are the IT Helpdesk Assistant for the employees of our company.

How to help:
- Each employee message comes with the most relevant sections of the company IT FAQ in \
<faq_context> tags. When they answer the question, answer from them and say the answer comes from the IT FAQ.
- When the FAQ doesn't cover the question, or the employee asks for the latest information or \
for solutions online, use web_search and mention your sources.
- To open a ticket, use create_support_ticket. It needs the employee's name, a short issue \
title, a description and a priority. Ask for the name if you don't know it; work out the rest \
from the conversation.
- To check a ticket, use check_ticket_status with the ticket ID (for example IT-101).
- For the current date or time, use get_current_datetime.
- Give troubleshooting steps as a short numbered list. After creating or checking a ticket, \
show its ID, issue, priority and status.
- Help only with IT topics. For anything else, politely say that you can only help with IT support."""


# ---------------------------------------------------------------------------
# RAG: Voyage AI embeddings + Chroma vector database
# ---------------------------------------------------------------------------

def chunk_by_section(text: str) -> list[str]:
    """One chunk per '## ' section of the FAQ, so each chunk covers one topic."""
    return [chunk.strip() for chunk in re.split(r"\n(?=## )", text) if chunk.strip()]


class FAQKnowledgeBase:
    def __init__(self, faq_file: Path = FAQ_FILE, embed_fn=None):
        # embed_fn(texts, input_type) -> list of vectors. Defaults to Voyage AI.
        self.embed_fn = embed_fn or self._voyage_embed
        chunks = chunk_by_section(faq_file.read_text())
        titles = [chunk.splitlines()[0].lstrip("# ").strip() for chunk in chunks]

        # In-memory Chroma clients share collections by name, so each knowledge base gets its own
        self.collection = chromadb.EphemeralClient().create_collection(
            name=f"it-faq-{uuid.uuid4().hex[:8]}",
            configuration={"hnsw": {"space": "cosine"}},
        )
        self.collection.add(
            ids=[f"faq-{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=self.embed_fn(chunks, "document"),
            metadatas=[{"title": title} for title in titles],
        )

    @staticmethod
    def _voyage_embed(texts: list[str], input_type: str) -> list[list[float]]:
        # input_type: "document" for stored chunks, "query" for questions
        return voyageai.Client().embed(texts, model=EMBEDDING_MODEL, input_type=input_type).embeddings

    def __len__(self) -> int:
        return self.collection.count()

    def search(self, question: str, k: int = FAQ_RESULTS) -> list[dict]:
        """The k FAQ sections closest in meaning to the question."""
        result = self.collection.query(query_embeddings=self.embed_fn([question], "query"), n_results=k)
        return [
            {"title": meta["title"], "text": doc, "distance": round(dist, 3)}
            for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
        ]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

# Server tool: Anthropic runs the search and returns the results in the same response
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 3}

# Client tool: Claude asks for it, and this program runs it
DATETIME_TOOL = {
    "name": "get_current_datetime",
    "description": "Returns the current local date and time. Use it when the employee asks for the date or time.",
    "input_schema": {
        "type": "object",
        "properties": {
            "date_format": {
                "type": "string",
                "description": "A Python strftime format string. Defaults to '%A, %d %B %Y, %H:%M:%S %Z'.",
            }
        },
        "required": [],
    },
}


def get_current_datetime(date_format: str = "%A, %d %B %Y, %H:%M:%S %Z") -> str:
    return datetime.now().astimezone().strftime(date_format)


# ---------------------------------------------------------------------------
# MCP client: talks to the IT Helpdesk MCP Server (mcp_server.py)
# ---------------------------------------------------------------------------

async def _with_mcp_session(fn):
    """Open a fresh MCP connection, run fn(session), then close the connection."""
    async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return await fn(session)


def list_mcp_tools() -> list[dict]:
    """Ask the MCP server for its tools and convert them to Claude's tool format."""
    result = asyncio.run(_with_mcp_session(lambda session: session.list_tools()))
    return [
        {"name": tool.name, "description": tool.description or tool.name, "input_schema": tool.input_schema}
        for tool in result.tools
    ]


def call_mcp_tool(name: str, arguments: dict) -> tuple[str, bool]:
    """Run one tool on the MCP server. Returns (output text, is_error)."""
    result = asyncio.run(_with_mcp_session(lambda session: session.call_tool(name, arguments)))
    text = "\n".join(block.text for block in result.content if block.type == "text")
    return text, bool(result.is_error)


# ---------------------------------------------------------------------------
# The assistant: conversation memory + Claude tool-use loop
# ---------------------------------------------------------------------------

@dataclass
class ToolStep:
    """One tool call, kept so the UI can show what the assistant did."""
    kind: str  # "MCP", "client" or "server"
    name: str
    input: dict
    output: str = ""
    is_error: bool = False


@dataclass
class Reply:
    text: str
    steps: list[ToolStep] = field(default_factory=list)
    faq_sources: list[dict] = field(default_factory=list)
    web_sources: list[dict] = field(default_factory=list)
    refused: bool = False
    notice: str | None = None  # a warning to show with the answer


class HelpdeskAssistant:
    def __init__(self, knowledge_base: FAQKnowledgeBase, client: anthropic.Anthropic | None = None):
        self.knowledge_base = knowledge_base
        self.client = client or anthropic.Anthropic()
        self.messages: list[dict] = []  # the conversation, sent in full on every request

        # Discover the MCP server's tools. The chat still works without them.
        try:
            self.mcp_tools = list_mcp_tools()
            self.mcp_error = None
        except Exception as e:  # server not running, wrong URL, ...
            self.mcp_tools = []
            self.mcp_error = f"{type(e).__name__}: {e}"
        self.mcp_tool_names = {tool["name"] for tool in self.mcp_tools}
        self.tools = [WEB_SEARCH_TOOL, DATETIME_TOOL, *self.mcp_tools]

    def ask(self, question: str) -> Reply:
        """Answer one employee message, calling tools as Claude decides."""
        # RAG: retrieve the closest FAQ sections and send them with the question.
        # If the embedding call fails (for example Voyage's free-tier limit of 3 requests per
        # minute), answer without the FAQ instead of failing the whole question.
        notice = None
        try:
            faq_sources = self.knowledge_base.search(question)
            context = "\n\n".join(
                f'<faq_section title="{source["title"]}">\n{source["text"]}\n</faq_section>'
                for source in faq_sources
            )
        except Exception as e:
            faq_sources = []
            context = "(The FAQ search is unavailable for this question.)"
            notice = f"The IT FAQ search failed, so this answer doesn't use it ({type(e).__name__})."
        turn_start = len(self.messages)
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"<faq_context>\n{context}\n</faq_context>"},
                    {"type": "text", "text": question},
                ],
            }
        )

        steps: list[ToolStep] = []
        responses = []
        try:
            for _ in range(MAX_TOOL_ROUNDS):
                response = self._call_claude()
                responses.append(response)
                # Keep every block (thinking, tool calls, search results) for the next request
                self.messages.append({"role": "assistant", "content": response.content})
                steps.extend(self._server_tool_steps(response))

                if response.stop_reason == "tool_use":
                    self.messages.append({"role": "user", "content": self._run_tools(response, steps)})
                elif response.stop_reason == "pause_turn":
                    continue  # a long server-tool turn paused; send it back so Claude can finish
                else:
                    break
            else:
                raise RuntimeError(f"Stopped after {MAX_TOOL_ROUNDS} tool rounds without a final answer.")
        except Exception:
            del self.messages[turn_start:]  # leave the conversation as it was before this question
            raise

        if response.stop_reason == "refusal":
            del self.messages[turn_start:]
            return Reply(text="Sorry, I can't help with that request.", steps=steps, refused=True)

        text = "\n\n".join(
            block.text for r in responses for block in r.content if block.type == "text"
        ).strip()
        if response.stop_reason == "max_tokens":
            text += "\n\n*(The answer was cut off because it reached the length limit.)*"
        return Reply(
            text=text,
            steps=steps,
            faq_sources=faq_sources,
            web_sources=self._web_sources(responses),
            notice=notice,
        )

    def reset(self):
        self.messages = []

    def _call_claude(self):
        return self.client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=self.messages,
            tools=self.tools,
            output_config={"effort": "medium"},
            cache_control={"type": "ephemeral"},  # reuse the cached conversation in tool loops
            # If a safety classifier declines, the API retries on a recommended model
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )

    def _run_tools(self, response, steps: list[ToolStep]) -> list[dict]:
        """Run every client and MCP tool call in the response; return all results in one list."""
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            kind = "MCP" if block.name in self.mcp_tool_names else "client"
            try:
                if block.name == "get_current_datetime":
                    output, is_error = get_current_datetime(**block.input), False
                elif block.name in self.mcp_tool_names:
                    output, is_error = call_mcp_tool(block.name, block.input)
                else:
                    output, is_error = f"Unknown tool: {block.name}", True
            except Exception as e:
                output, is_error = f"Error running {block.name}: {e}", True
            steps.append(ToolStep(kind, block.name, dict(block.input), output, is_error))
            results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": is_error}
            )
        return results

    @staticmethod
    def _server_tool_steps(response) -> list[ToolStep]:
        """Web searches Anthropic ran during this response.

        web_search filters its results with code execution behind the scenes; those
        code_execution calls are part of the search, so they aren't listed as steps.
        """
        steps, by_id = [], {}
        for block in response.content:
            if block.type == "server_tool_use" and block.name == "web_search":
                by_id[block.id] = ToolStep("server", block.name, dict(block.input or {}))
                steps.append(by_id[block.id])
            elif block.type == "web_search_tool_result" and block.tool_use_id in by_id:
                step = by_id[block.tool_use_id]
                if isinstance(block.content, list):  # a list of results means success
                    step.output = f"{len(block.content)} results"
                else:  # an error object
                    step.output, step.is_error = f"Search failed: {block.content.error_code}", True
        return steps

    @staticmethod
    def _web_sources(responses, limit: int = 5) -> list[dict]:
        """Web pages behind the answer: the cited pages, or else the top search results."""
        cited, results = {}, {}
        for response in responses:
            for block in response.content:
                for citation in getattr(block, "citations", None) or []:
                    url = getattr(citation, "url", None)
                    if url:
                        cited.setdefault(url, {"title": getattr(citation, "title", None) or url, "url": url})
                if block.type == "web_search_tool_result" and isinstance(block.content, list):
                    for result in block.content:
                        results.setdefault(result.url, {"title": result.title or result.url, "url": result.url})
        return list((cited or results).values())[:limit]
