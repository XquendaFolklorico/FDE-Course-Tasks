"""IT Helpdesk MCP Server.

Exposes two tools over MCP (Streamable HTTP):
  - create_support_ticket: create a new IT support ticket
  - check_ticket_status:   look up an existing ticket by ID

Tickets are stored in an in-memory Python dictionary, seeded from mcp_tool_data.json
(IT-101 to IT-103). New tickets last until the server stops.

Run:  python mcp_server.py   -> serves http://localhost:3000/mcp
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from mcp.server import MCPServer
from pydantic import Field

SEED_FILE = Path(__file__).resolve().parent / "mcp_tool_data.json"
HOST, PORT = "127.0.0.1", 3000

mcp_server = MCPServer(name="IT Helpdesk MCP Server", version="1.0.0")

# In-memory ticket store: ticket ID -> ticket details
TICKETS: dict[str, dict] = {
    ticket["ticket_id"]: ticket for ticket in json.loads(SEED_FILE.read_text())
}


def next_ticket_id() -> str:
    """IT-101, IT-102, ... -> the next free number."""
    numbers = [int(ticket_id.split("-")[1]) for ticket_id in TICKETS]
    return f"IT-{max(numbers, default=100) + 1}"


@mcp_server.tool(
    name="create_support_ticket",
    description=(
        "Create a new IT support ticket for an employee and return the saved ticket, "
        "including its new ticket ID and status. Ask the employee for their name if you don't know it."
    ),
)
def create_support_ticket(
    employee: Annotated[str, Field(description="Name of the employee reporting the problem.")],
    issue: Annotated[str, Field(description='Short title of the problem, for example "Laptop running slowly".')],
    description: Annotated[str, Field(description="Details of the problem in one or two sentences.")],
    priority: Annotated[
        Literal["Low", "Medium", "High"],
        Field(description="How much the problem blocks the employee's work."),
    ] = "Medium",
) -> str:
    ticket = {
        "ticket_id": next_ticket_id(),
        "employee": employee,
        "issue": issue,
        "description": description,
        "priority": priority,
        "status": "Open",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    TICKETS[ticket["ticket_id"]] = ticket
    return json.dumps(ticket)


@mcp_server.tool(
    name="check_ticket_status",
    description="Look up an existing IT support ticket by its ID (for example IT-101) and return its details and status.",
)
def check_ticket_status(
    ticket_id: Annotated[str, Field(description="The ticket ID, for example IT-101.")],
) -> str:
    ticket =TICKETS.get(ticket_id.strip().upper())
    if ticket is None:
        known = ", ".join(sorted(TICKETS))
        return json.dumps({"error": f"No ticket found with ID {ticket_id}. Existing tickets: {known}"})
    return json.dumps(ticket)


if __name__ == "__main__":
    print(f"IT Helpdesk MCP Server running at http://localhost:{PORT}/mcp")
    mcp_server.run(transport="streamable-http", host=HOST, port=PORT)
