"""trader MCP server — project operations exposed as MCP tools.

Tool catalog, safety tiers, and build phasing are designed in the vault note
"BNB Hackathon/MCP Server.md". This is a skeleton; tools are added per phase.
"""

from trader.mcp_server.server import mcp

__all__ = ["mcp"]
