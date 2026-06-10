"""trader.experiment — the laptop-side RL experiment-loop surface.

The importable cores the MCP server's RL tools wrap (vault "MCP Server" §"Training / RL
experiment loop"): the disciplined SSH helper (`remote`), the published-bundle diagnostics
(`diagnostics`), and the committed champion/ledger (`champion`). Kept free of FastMCP so each
piece stays unit-testable; the tools in `trader.mcp_server.server` are thin wrappers over these.
"""
