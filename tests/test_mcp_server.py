import asyncio
import io
import shutil
from pathlib import Path
from uuid import uuid4

from memk.mcp.server import (
    TOOLS,
    _call_tool,
    _ensure_workspace_id,
    _format_results,
    handle_request,
    read_message,
    write_message,
)


def test_mcp_initialize_response():
    response = asyncio.run(handle_request({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {},
    }))

    assert response["id"] == 1
    assert response["result"]["serverInfo"]["name"] == "memorykernel"
    assert "tools" in response["result"]["capabilities"]


def test_mcp_tools_list_exposes_starter_tools():
    response = asyncio.run(handle_request({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
    }))

    names = {tool["name"] for tool in response["result"]["tools"]}

    assert names == {
        "memk_guide",
        "memk_remember",
        "memk_recall",
        "memk_context",
        "memk_health",
    }
    remember_tool = next(tool for tool in TOOLS if tool["name"] == "memk_remember")
    assert remember_tool["inputSchema"]["required"] == ["content"]


def test_mcp_unknown_tool_returns_json_rpc_error():
    response = asyncio.run(handle_request({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "memk_missing", "arguments": {}},
    }))

    assert response["id"] == 3
    assert response["error"]["code"] == -32000
    assert "Unknown tool" in response["error"]["message"]


def test_mcp_framing_roundtrip():
    stream = io.BytesIO()
    write_message(stream, {"jsonrpc": "2.0", "id": 4, "result": {"ok": True}})
    stream.seek(0)

    message = read_message(stream)

    assert message["id"] == 4
    assert message["result"]["ok"] is True


def test_mcp_workspace_auto_init_with_explicit_workspace_id(monkeypatch):
    workspace = Path.cwd() / f".pytest-mcp-workspace-{uuid4().hex}"
    workspace.mkdir()
    (workspace / ".git").mkdir()

    try:
        monkeypatch.chdir(workspace)

        workspace_id = _ensure_workspace_id("agent-workspace")

        assert workspace_id == "agent-workspace"
        assert (workspace / ".memk" / "manifest.json").exists()
        assert (workspace / ".memk" / "state" / "state.db").exists()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_mcp_guide_returns_agent_usage_policy():
    result = asyncio.run(_call_tool("memk_guide", {}))

    text = result["content"][0]["text"]
    assert "Use memk_remember after durable facts" in text
    assert "Avoid storing temporary actions" in text


def test_mcp_health_does_not_require_service_runtime(monkeypatch):
    workspace = Path.cwd() / f".pytest-mcp-workspace-{uuid4().hex}"
    workspace.mkdir()
    (workspace / ".git").mkdir()

    def fail_service():
        raise AssertionError("health should not load the service runtime")

    try:
        monkeypatch.chdir(workspace)
        monkeypatch.setattr("memk.mcp.server._service_instance", fail_service)

        result = asyncio.run(_call_tool("memk_health", {}))

        text = result["content"][0]["text"]
        assert "Grade:" in text
        assert "Memories: 0" in text
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_format_results_for_agent_text():
    text = _format_results([
        {
            "item_type": "memory",
            "content": "The frontend uses Vite",
            "score": 0.8123,
        }
    ])

    assert "1. [memory] 0.812 - The frontend uses Vite" in text
