"""Smoke test — confirms the package imports and the MCP server skeleton is wired."""

from trader.mcp_server.server import health_status


def test_health_status():
    s = health_status()
    assert s["status"] == "ok"
    assert s["server"] == "trader"


def test_mcp_object_exists():
    from trader.mcp_server import mcp

    assert mcp is not None
