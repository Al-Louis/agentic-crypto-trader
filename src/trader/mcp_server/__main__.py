"""Entry point: `python -m trader.mcp_server` (stdio MCP server)."""

from trader.mcp_server.server import mcp


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
