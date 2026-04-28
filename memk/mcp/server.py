"""
Minimal MCP stdio server for MemoryKernel.

The server exposes the product-first tool surface:

- memk_remember
- memk_recall
- memk_context
- memk_health

It uses JSON-RPC over MCP's Content-Length framed stdio transport.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, BinaryIO, Optional

from memk import __version__
from memk.core.service import MemoryKernelService


SERVER_NAME = "memorykernel"
PROTOCOL_VERSION = "2024-11-05"

_service: Optional[MemoryKernelService] = None


TOOLS: list[dict[str, Any]] = [
    {
        "name": "memk_remember",
        "description": "Store a project memory for future AI agent sessions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Memory content to store."},
                "importance": {
                    "type": "number",
                    "description": "Memory priority from 0.0 to 1.0.",
                    "default": 0.5,
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence from 0.0 to 1.0.",
                    "default": 1.0,
                },
                "workspace_id": {"type": "string", "description": "Optional workspace scope."},
            },
            "required": ["content"],
        },
    },
    {
        "name": "memk_recall",
        "description": "Recall relevant project memories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Recall query."},
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of memories to return.",
                    "default": 5,
                },
                "workspace_id": {"type": "string", "description": "Optional workspace scope."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memk_context",
        "description": "Build compact agent context from project memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Context query."},
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum context size.",
                    "default": 1200,
                },
                "threshold": {
                    "type": "number",
                    "description": "Minimum relevance score.",
                    "default": 0.3,
                },
                "workspace_id": {"type": "string", "description": "Optional workspace scope."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memk_health",
        "description": "Show project memory health and next actions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "workspace_id": {"type": "string", "description": "Optional workspace scope."}
            },
        },
    },
]


def _service_instance() -> MemoryKernelService:
    global _service
    if _service is None:
        _service = MemoryKernelService(allow_direct_writes=True)
    return _service


def _ensure_workspace_id(workspace_id: Optional[str] = None) -> str:
    from memk.storage.db import MemoryDB
    from memk.workspace.manager import WorkspaceManager

    ws = WorkspaceManager()
    if not ws.is_initialized():
        ws.init_workspace()
        MemoryDB(ws.get_db_path()).init_db()
    if workspace_id:
        return workspace_id
    return ws.get_manifest().brain_id


def _text_result(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _format_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No relevant memories found."

    lines = []
    for index, item in enumerate(results, start=1):
        score = float(item.get("score", 0.0))
        content = str(item.get("content", ""))
        item_type = str(item.get("item_type", "memory"))
        lines.append(f"{index}. [{item_type}] {score:.3f} - {content}")
    return "\n".join(lines)


def _format_health(diag: dict[str, Any], workspace_id: str) -> str:
    stats = diag.get("db_stats", {})
    runtime = diag.get("runtime", {})
    total_memories = int(stats.get("total_memories", 0) or 0)
    total_facts = int(stats.get("total_active_facts", 0) or 0)
    total_items = total_memories + total_facts
    embedded = int(stats.get("embedded_memories", 0) or 0) + int(stats.get("embedded_facts", 0) or 0)
    embed_pct = (embedded / total_items * 100) if total_items else 100.0

    if total_items == 0:
        grade = "B"
        next_action = "Store the first project memory with memk_remember."
    elif embed_pct >= 90:
        grade = "A"
        next_action = "Memory looks healthy."
    else:
        grade = "C"
        next_action = "Run memk health or memk doctor locally to inspect embedding coverage."

    return "\n".join(
        [
            f"Grade: {grade}",
            f"Workspace: {workspace_id}",
            f"Memories: {total_memories}",
            f"Facts: {total_facts}",
            f"Embedded: {embedded}/{total_items} ({embed_pct:.1f}%)",
            f"Database MB: {float(stats.get('database_size_mb', 0) or 0):.2f}",
            f"Index entries: {runtime.get('index_entries', 0)}",
            f"Next action: {next_action}",
        ]
    )


async def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name not in {tool["name"] for tool in TOOLS}:
        raise ValueError(f"Unknown tool: {name}")

    service = _service_instance()
    workspace_id = _ensure_workspace_id(arguments.get("workspace_id"))

    if name == "memk_remember":
        result = await service.add_memory(
            str(arguments["content"]),
            float(arguments.get("importance", 0.5)),
            float(arguments.get("confidence", 1.0)),
            workspace_id,
        )
        return _text_result(f"Stored memory {result['id']} in workspace {workspace_id}.")

    if name == "memk_recall":
        result = await service.search(
            str(arguments["query"]),
            int(arguments.get("limit", 5)),
            workspace_id,
        )
        return _text_result(_format_results(result.get("results", [])))

    if name == "memk_context":
        result = await service.build_context(
            str(arguments["query"]),
            int(arguments.get("max_chars", 1200)),
            float(arguments.get("threshold", 0.3)),
            workspace_id,
        )
        return _text_result(result.get("context", ""))

    if name == "memk_health":
        diag = service.get_diagnostics(workspace_id)
        return _text_result(_format_health(diag, workspace_id))

    raise ValueError(f"Unknown tool: {name}")


async def handle_request(message: dict[str, Any]) -> Optional[dict[str, Any]]:
    method = message.get("method")
    request_id = message.get("id")

    if method == "notifications/initialized":
        return None

    try:
        if method == "initialize":
            result = {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": __version__},
            }
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            params = message.get("params", {})
            result = await _call_tool(str(params.get("name")), params.get("arguments", {}) or {})
        elif method == "ping":
            result = {}
        else:
            raise ValueError(f"Unsupported method: {method}")

        if request_id is None:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except Exception as exc:
        if request_id is None:
            return None
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32000, "message": str(exc)},
        }


def read_message(stream: BinaryIO) -> Optional[dict[str, Any]]:
    headers: dict[str, str] = {}

    while True:
        line = stream.readline()
        if line == b"":
            return None
        line = line.strip()
        if not line:
            break
        key, _, value = line.decode("ascii").partition(":")
        headers[key.lower()] = value.strip()

    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    body = stream.read(length)
    return json.loads(body.decode("utf-8"))


def write_message(stream: BinaryIO, message: dict[str, Any]) -> None:
    body = json.dumps(message, separators=(",", ":")).encode("utf-8")
    stream.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    stream.write(body)
    stream.flush()


def main() -> None:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("MemoryKernel MCP server. Configure your MCP client to run: memk-mcp")
        return

    while True:
        message = read_message(sys.stdin.buffer)
        if message is None:
            break
        response = asyncio.run(handle_request(message))
        if response is not None:
            write_message(sys.stdout.buffer, response)


if __name__ == "__main__":
    main()
