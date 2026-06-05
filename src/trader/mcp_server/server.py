"""trader MCP server (skeleton).

Exposes project operations as MCP tools per the design in the vault note
"BNB Hackathon/MCP Server.md". Phase-1 skeleton: only `health` is real; domain tools are
stubs added per phase. Pure helpers back each tool so they stay unit-testable.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("trader")

PHASE = "skeleton"


def health_status() -> dict:
    """Core health logic (pure, testable)."""
    return {"status": "ok", "server": "trader", "phase": PHASE}


@mcp.tool()
def health() -> dict:
    """Health check — confirm the trader MCP server is alive and report its build phase."""
    return health_status()


@mcp.tool()
def eligible_tokens() -> dict:
    """[STUB] Fixed competition token universe + metadata.

    Not implemented in the skeleton. Design: vault "MCP Server" (Data tier, Phase 2).
    """
    return {"status": "not_implemented", "see": "BNB Hackathon/MCP Server.md"}
