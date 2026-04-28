import typer
import requests
import time
import datetime
import logging
import os
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from memk.server.manager import URL, is_running
from memk.core.service import MemoryKernelService

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="MemoryKernel (memk) - Project memory that AI agents can carry across sessions",
    no_args_is_help=True
)
console = Console()
_service_instance = None

def get_service() -> MemoryKernelService:
    global _service_instance
    if _service_instance is None:
        _service_instance = MemoryKernelService(allow_direct_writes=True)
    return _service_instance

def get_workspace_id() -> str:
    """Resolve the brain ID from the local manifest, creating it on first use."""
    ws = _ensure_workspace(auto_create=True, announce=False)
    try:
        return ws.get_manifest().brain_id
    except Exception:
        return "default"

def _ensure_workspace(auto_create: bool = True, announce: bool = False):
    """Return the workspace manager and optionally initialize first-run state."""
    from memk.workspace.manager import WorkspaceManager
    from memk.storage.db import MemoryDB

    ws = WorkspaceManager()
    if ws.is_initialized() or not auto_create:
        return ws

    try:
        ws.init_workspace()
        MemoryDB(ws.get_db_path()).init_db()
        if announce:
            console.print("[green]Initialized MemoryKernel workspace.[/green]")
            console.print(f"Path: [dim]{ws.memk_path}[/dim]")
        return ws
    except Exception as e:
        raise RuntimeError(f"Failed to initialize workspace: {e}") from e

def _post_v1(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST to the versioned daemon API and return decoded JSON."""
    resp = requests.post(
        f"{URL}/v1/{endpoint.lstrip('/')}",
        json=payload,
        timeout=30,
        headers=_daemon_headers(),
    )
    _raise_for_status(resp)
    return resp.json()

def _daemon_headers() -> Dict[str, str]:
    """Return daemon auth headers when MEMK_API_TOKEN is configured."""
    token = os.getenv("MEMK_API_TOKEN", "").strip()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}

def _raise_for_status(resp: requests.Response) -> None:
    """Raise a readable daemon error, including v1 error codes when present."""
    if resp.status_code < 400:
        return
    message = resp.text
    try:
        detail = resp.json().get("detail")
        if isinstance(detail, dict) and detail.get("code"):
            message = f"{detail['code']}: {detail.get('message', '')}".strip()
    except Exception:
        pass
    raise RuntimeError(message)

def _response_data(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Handle both v1 APIResponse and older flat daemon responses."""
    data = resp.get("data")
    return data if isinstance(data, dict) else resp

def _health_grade(initialized: bool, total_items: int, embed_pct: float, daemon_running: bool) -> str:
    """Return a simple user-facing health grade."""
    if not initialized:
        return "D"
    if total_items == 0:
        return "B"
    if embed_pct >= 90 and daemon_running:
        return "A"
    if embed_pct >= 70:
        return "B"
    return "C"

def _render_health(
    *,
    initialized: bool,
    daemon_running: bool,
    workspace_id: str,
    root: str,
    stats: Dict[str, Any],
    runtime: Optional[Dict[str, Any]] = None,
) -> None:
    """Render a compact visual health view."""
    total_memories = int(stats.get("total_memories", 0) or 0)
    total_facts = int(stats.get("total_active_facts", 0) or 0)
    total_items = total_memories + total_facts
    total_embedded = int(stats.get("embedded_memories", 0) or 0) + int(stats.get("embedded_facts", 0) or 0)
    embed_pct = (total_embedded / total_items * 100) if total_items else 100.0
    grade = _health_grade(initialized, total_items, embed_pct, daemon_running)
    grade_color = "green" if grade in {"A", "B"} else "yellow"

    lines = [
        f"Grade: [{grade_color}]{grade}[/{grade_color}]",
        f"Profile: [cyan]{stats.get('performance_profile', 'lite')}[/cyan] ({stats.get('index_mode', 'sqlite')})",
        f"Workspace: [cyan]{root}[/cyan]",
        f"Brain ID: [magenta]{workspace_id[:12]}[/magenta]",
        f"Daemon: [{'green' if daemon_running else 'yellow'}]{'running' if daemon_running else 'not running'}[/]",
        f"Memories: [cyan]{total_memories}[/cyan]",
        f"Facts: [cyan]{total_facts}[/cyan]",
        f"Embedded: [cyan]{total_embedded}/{total_items}[/cyan] ({embed_pct:.1f}%)",
        f"FTS: [{'green' if stats.get('fts_available', False) else 'yellow'}]{'available' if stats.get('fts_available', False) else 'fallback'}[/]",
        f"Database: [cyan]{float(stats.get('database_size_mb', 0) or 0):.2f} MB[/cyan]",
    ]
    if runtime:
        lines.append(f"Index entries: [cyan]{runtime.get('index_entries', 0)}[/cyan]")
        lines.append(f"Active jobs: [cyan]{runtime.get('active_jobs', 0)}[/cyan]")

    advice = []
    if not initialized:
        advice.append("Run `memk remember \"...\"` to create the first memory.")
    elif total_items == 0:
        advice.append("Store your first durable project fact with `memk remember`.")
    if not daemon_running:
        advice.append("Run `memk serve` for repeated agent/SDK use.")
    if total_items and embed_pct < 90:
        advice.append("Run `memk doctor` to inspect low embedding coverage.")

    if advice:
        lines.append("")
        lines.append("[bold]Next actions[/bold]")
        lines.extend(f"- {item}" for item in advice)

    console.print(Panel("\n".join(lines), title="MemoryKernel Health", expand=False))

def _setup_instructions(tool: str) -> str:
    """Return copy/paste setup instructions for a supported AI tool."""
    normalized = tool.lower().replace("-", "").replace("_", "")
    if normalized in {"mcp", "generic"}:
        return "\n".join([
            "Generic MCP client",
            "",
            "Server command:",
            "  memk-mcp",
            "",
            "Config shape:",
            '{',
            '  "mcpServers": {',
            '    "memorykernel": {',
            '      "command": "memk-mcp",',
            '      "args": []',
            '    }',
            '  }',
            '}',
        ])
    if normalized in {"claude", "claudecode"}:
        return "\n".join([
            "Claude Code",
            "",
            "Run:",
            "  claude mcp add --transport stdio memorykernel --scope user -- memk-mcp",
            "",
            "Then ask Claude:",
            '  Remember: the billing service owns invoice numbering.',
            '  Recall what we know about billing.',
        ])
    if normalized == "cursor":
        return "\n".join([
            "Cursor",
            "",
            "Add to ~/.cursor/mcp.json or .cursor/mcp.json:",
            '{',
            '  "mcpServers": {',
            '    "memorykernel": {',
            '      "command": "memk-mcp",',
            '      "args": []',
            '    }',
            '  }',
            '}',
            "",
            "Restart Cursor after changing MCP config.",
        ])
    if normalized in {"vscode", "vs"}:
        return "\n".join([
            "VS Code",
            "",
            "Add to .vscode/mcp.json or your user MCP config:",
            '{',
            '  "servers": {',
            '    "memorykernel": {',
            '      "type": "stdio",',
            '      "command": "memk-mcp",',
            '      "args": []',
            '    }',
            '  }',
            '}',
        ])
    if normalized == "openclaw":
        return "\n".join([
            "OpenClaw",
            "",
            "Run:",
            '  openclaw mcp set memorykernel \'{"command":"memk-mcp"}\'',
            "",
            "MemoryKernel runs as a local stdio MCP server.",
        ])

    supported = "claude, cursor, vscode, openclaw, mcp"
    raise ValueError(f"Unknown tool '{tool}'. Supported tools: {supported}")

def _guide_text() -> str:
    """Return first-run guidance for humans and agents."""
    return "\n".join([
        "MemoryKernel helps AI agents keep project knowledge across sessions.",
        "",
        "Use these first:",
        '  1. memk remember "Decision: use PostgreSQL for billing writes"',
        '  2. memk recall "billing database decision"',
        '  3. memk health',
        "",
        "Store durable facts:",
        "  - decisions, conventions, bug causes, fixes, preferences, workflows",
        "",
        "Avoid temporary notes:",
        "  - opened a file, ran a command, read a README",
        "",
        "Connect an agent:",
        "  memk setup claude    # or cursor, vscode, openclaw",
        "",
        "Optional semantic model:",
        '  python -m pip install -e ".[semantic]"',
        "  Set MEMK_EMBEDDER=hashing for fastest deterministic startup.",
    ])

def _add_memory(content: str, importance: float, confidence: float, workspace: Optional[str]) -> None:
    """Shared implementation for memory write commands."""
    _ensure_workspace(auto_create=True, announce=True)
    workspace_id = workspace or get_workspace_id()
    try:
        if is_running():
            resp = _post_v1("remember", {
                "content": content, "importance": importance, "confidence": confidence, "workspace_id": workspace_id
            })
            data = _response_data(resp)
            console.print(f"[green]Added via Daemon![/green] ID: [cyan]{data['id'][:8]}...[/cyan] (WS: {workspace_id[:8]}...)")
            return

        service = get_service()
        import asyncio
        res = asyncio.run(service.add_memory(content, importance, confidence, workspace_id))
        console.print(f"[green]Added memory successfully![/green] ID: [cyan]{res['id'][:8]}...[/cyan]")
    except Exception as e:
        console.print(f"[bold red]Failed to add memory:[/bold red] {e}")
        raise typer.Exit(code=1)

# --- Lifecycle Commands ---

@app.command()
def serve():
    """Start the persistent MemoryKernel daemon."""
    from memk.server import manager
    manager.start()

@app.command()
def stop():
    """Stop the running MemoryKernel daemon."""
    from memk.server import manager
    manager.stop()

@app.command()
def status():
    """Check the status of the local daemon and workspace."""
    from memk.workspace.manager import WorkspaceManager
    from memk.server import manager as server_manager
    
    # 1. Workspace Status
    ws = WorkspaceManager()
    info = ws.get_status_info()
    
    console.print("[bold blue]📂 Workspace Status[/bold blue]")
    console.print(f"  Root: [cyan]{info['root']}[/cyan]")
    console.print(f"  Initialized: [{'green' if info['initialized'] else 'red'}]{info['initialized']}[/{'green' if info['initialized'] else 'red'}]")
    
    if info['initialized']:
        console.print(f"  Brain ID: [magenta]{info['brain_id']}[/magenta]")
        console.print(f"  Generation: [yellow]{info['generation']}[/yellow]")
    
    console.print(f"\n[bold blue]🤖 Daemon Status[/bold blue]")
    stat = server_manager.get_status()
    color = "green" if "RUNNING" in stat else "red"
    console.print(f"  Status: [{color}]{stat}[/{color}]")
    console.print(f"  Endpoint: [cyan]{server_manager.URL}[/cyan]")

@app.command()
def health():
    """Show visual health for the current project memory."""
    try:
        ws = _ensure_workspace(auto_create=True, announce=True)
        workspace_id = get_workspace_id()
        daemon_running = is_running()

        if daemon_running:
            status_resp = requests.get(
                f"{URL}/v1/status",
                params={"workspace_id": workspace_id},
                timeout=10,
                headers=_daemon_headers(),
            )
            _raise_for_status(status_resp)
            data = _response_data(status_resp.json())
            _render_health(
                initialized=bool(data.get("initialized", True)),
                daemon_running=True,
                workspace_id=data.get("workspace_id", workspace_id),
                root=data.get("workspace_root", str(ws.root)),
                stats=data.get("stats", {}),
            )
            return

        from memk.storage.db import MemoryDB
        db = MemoryDB(ws.get_db_path())
        db.init_db()
        stats = db.get_stats()
        _render_health(
            initialized=ws.is_initialized(),
            daemon_running=False,
            workspace_id=workspace_id,
            root=str(ws.root),
            stats=stats,
            runtime={"index_entries": "not loaded", "active_jobs": 0},
        )
    except Exception as e:
        console.print(f"[bold red]Health check failed:[/bold red] {e}")
        raise typer.Exit(code=1)

@app.command()
def setup(
    tool: str = typer.Argument(
        "mcp",
        help="Tool to configure: claude, cursor, vscode, openclaw, or mcp.",
    )
):
    """Print copy/paste setup instructions for AI tools."""
    try:
        console.print(Panel(_setup_instructions(tool), title="MemoryKernel Setup", expand=False))
    except Exception as e:
        console.print(f"[bold red]Setup failed:[/bold red] {e}")
        raise typer.Exit(code=1)

@app.command()
def guide():
    """Show the shortest path from install to useful project memory."""
    console.print(Panel(_guide_text(), title="MemoryKernel Guide", expand=False))

@app.command()
def stats(workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace scope.")):
    """Show daemon metrics through the versioned API."""
    workspace_id = workspace or get_workspace_id()
    if not is_running():
        console.print("[red]Daemon not running.[/red]")
        raise typer.Exit(code=1)

    try:
        resp = requests.get(
            f"{URL}/v1/metrics",
            params={"workspace_id": workspace_id},
            timeout=10,
            headers=_daemon_headers(),
        )
        _raise_for_status(resp)
        data = _response_data(resp.json())

        console.print("[bold blue]MemoryKernel Metrics[/bold blue]")
        requests_data = data.get("requests", {})
        latency = data.get("latency", {})
        errors = data.get("errors", {})
        database = data.get("database", {})

        console.print(f"  Requests: [cyan]{requests_data.get('total', 0)}[/cyan]")
        console.print(f"  Request Rate: [cyan]{requests_data.get('rate_per_sec', 0)} / sec[/cyan]")
        console.print(f"  Error Rate: [cyan]{errors.get('rate', 0)}[/cyan]")
        console.print(f"  Latency p95: [cyan]{latency.get('p95', 0)} ms[/cyan]")
        console.print(f"  DB Size: [cyan]{database.get('size_mb', 0)} MB[/cyan]")
        console.print(f"  Memories: [cyan]{database.get('total_memories', 0)}[/cyan]")
        console.print(f"  Facts: [cyan]{database.get('total_facts', 0)}[/cyan]")
    except Exception as e:
        console.print(f"[bold red]Stats failed:[/bold red] {e}")
        raise typer.Exit(code=1)

@app.command()
def backup(
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Backup archive path.")
):
    """Create a zip backup of the current workspace memory store."""
    from pathlib import Path
    import zipfile
    from memk.workspace.manager import WorkspaceManager

    ws = WorkspaceManager()
    if not ws.is_initialized():
        console.print("[red]Workspace not initialized. Run 'memk init' first.[/red]")
        raise typer.Exit(code=1)

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = Path(output) if output else ws.root / f"memk-backup-{stamp}.zip"
    archive = archive.expanduser().resolve()
    if archive.exists():
        console.print(f"[red]Backup already exists:[/red] {archive}")
        raise typer.Exit(code=1)

    archive.parent.mkdir(parents=True, exist_ok=True)
    db_path = Path(ws.get_db_path())
    files = [
        (ws.manifest_path, "manifest.json"),
        (db_path, "state/state.db"),
        (Path(str(db_path) + "-wal"), "state/state.db-wal"),
        (Path(str(db_path) + "-shm"), "state/state.db-shm"),
    ]

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, arcname in files:
            if path.exists():
                zf.write(path, arcname)

    console.print(f"[green]Backup created:[/green] [cyan]{archive}[/cyan]")

@app.command()
def restore(
    archive: str,
    force: bool = typer.Option(False, "--force", help="Confirm replacing current .memk state.")
):
    """Restore a workspace memory store from a backup archive."""
    from pathlib import Path, PurePosixPath
    import shutil
    import zipfile
    from memk.workspace.manager import WorkspaceManager

    if is_running():
        console.print("[red]Stop the daemon before restoring: memk stop[/red]")
        raise typer.Exit(code=1)

    if not force:
        console.print("[red]Restore replaces local .memk state. Re-run with --force to confirm.[/red]")
        raise typer.Exit(code=1)

    archive_path = Path(archive).expanduser().resolve()
    if not archive_path.exists():
        console.print(f"[red]Backup not found:[/red] {archive_path}")
        raise typer.Exit(code=1)

    ws = WorkspaceManager()
    allowed = {
        "manifest.json",
        "state/state.db",
        "state/state.db-wal",
        "state/state.db-shm",
    }

    with zipfile.ZipFile(archive_path, "r") as zf:
        names = {info.filename.replace("\\", "/") for info in zf.infolist()}
        if "manifest.json" not in names or "state/state.db" not in names:
            console.print("[red]Invalid backup: manifest.json and state/state.db are required.[/red]")
            raise typer.Exit(code=1)

        ws.memk_path.mkdir(parents=True, exist_ok=True)
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            parts = PurePosixPath(name).parts
            if name not in allowed or ".." in parts or PurePosixPath(name).is_absolute():
                console.print(f"[red]Invalid backup entry:[/red] {name}")
                raise typer.Exit(code=1)

            target = (ws.memk_path / Path(*parts)).resolve()
            try:
                target.relative_to(ws.memk_path.resolve())
            except ValueError:
                console.print(f"[red]Invalid backup entry:[/red] {name}")
                raise typer.Exit(code=1)

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)

    console.print(f"[green]Backup restored from:[/green] [cyan]{archive_path}[/cyan]")

# --- Core Commands ---

@app.command()
def init():
    """Initialize the MemoryKernel workspace and brain state."""
    from memk.workspace.manager import WorkspaceManager
    from memk.storage.db import MemoryDB
    try:
        ws = WorkspaceManager()
        is_reinit = ws.is_initialized()
        
        manifest = ws.init_workspace()
        MemoryDB(ws.get_db_path()).init_db()
        
        msg = "re-initialized" if is_reinit else "initialized"
        console.print(f"[green]MemoryKernel workspace {msg} successfully.[/green]")
        console.print(f"Brain ID: [magenta]{manifest.brain_id}[/magenta]")
        console.print(f"Path: [dim]{ws.memk_path}[/dim]")
        console.print()
        console.print(Panel(_guide_text(), title="Next Steps", expand=False))
    except Exception as e:
        console.print(f"[bold red]Failed to initialize workspace:[/bold red] {e}")
        raise typer.Exit(code=1)

@app.command()
def add(
    content: str,
    importance: float = typer.Option(0.5, "--importance", "-i", min=0, max=1, help="Priority of this memory."),
    confidence: float = typer.Option(1.0, "--confidence", "-c", min=0, max=1, help="Certainty of this memory."),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace scope.")
):
    """Remember something the project agent should retain."""
    _add_memory(content, importance, confidence, workspace)

@app.command("remember")
def remember(
    content: str,
    importance: float = typer.Option(0.5, "--importance", "-i", min=0, max=1, help="Priority of this memory."),
    confidence: float = typer.Option(1.0, "--confidence", "-c", min=0, max=1, help="Certainty of this memory."),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace scope.")
):
    """Remember something the project agent should retain."""
    _add_memory(content, importance, confidence, workspace)

def _search_results(query: str, limit: int, workspace: Optional[str]) -> List[Dict[str, Any]]:
    """Shared implementation for recall/search commands."""
    _ensure_workspace(auto_create=True, announce=False)
    workspace_id = workspace or get_workspace_id()
    if is_running():
        resp = _post_v1("search", {"query": query, "limit": limit, "workspace_id": workspace_id})
        return _response_data(resp).get("results", [])

    service = get_service()
    import asyncio
    resp = asyncio.run(service.search(query, limit, workspace_id))
    return resp.get("results", [])

def _render_search_results(query: str, results: List[Dict[str, Any]]) -> None:
    table = Table(title=f"Memory Recall: '{query}'")
    table.add_column("Type", style="dim")
    table.add_column("Content", style="cyan")
    table.add_column("Score", justify="right")

    for r in results:
        table.add_row(r["item_type"], r["content"], f"{r['score']:.3f}")

    console.print(table)

@app.command()
def search(
    query: str,
    limit: int = typer.Option(10, "--limit", "-l", help="Number of results."),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace scope.")
):
    """Search project memory."""
    try:
        _render_search_results(query, _search_results(query, limit, workspace))
    except Exception as e:
        console.print(f"[bold red]Search failed:[/bold red] {e}")
        raise typer.Exit(code=1)

@app.command("recall")
def recall(
    query: str,
    limit: int = typer.Option(10, "--limit", "-l", help="Number of results."),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace scope.")
):
    """Recall project memory for an AI agent or developer."""
    try:
        _render_search_results(query, _search_results(query, limit, workspace))
    except Exception as e:
        console.print(f"[bold red]Recall failed:[/bold red] {e}")
        raise typer.Exit(code=1)

@app.command()
def context(
    query: str,
    max_chars: int = typer.Option(500, "--limit", "-l", help="Max window size."),
    threshold: float = typer.Option(0.3, "--threshold", "-t", help="Relevance threshold."),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace scope.")
):
    """Compile optimized RAG context for AI agents."""
    _ensure_workspace(auto_create=True, announce=False)
    workspace_id = workspace or get_workspace_id()
    try:
        if is_running():
            resp = _post_v1("context", {
                "query": query, "max_chars": max_chars, "threshold": threshold, "workspace_id": workspace_id
            })
            ctx = _response_data(resp).get("context", "")
        else:
            service = get_service()
            import asyncio
            resp = asyncio.run(service.build_context(query, max_chars, threshold, workspace_id))
            ctx = resp.get("context", "")

        console.print("\n[bold]Generated Context:[/bold]")
        console.print(f"[dim]{'-' * 40}[/dim]")
        console.print(ctx)
        console.print(f"[dim]{'-' * 40}[/dim]")
    except Exception as e:
        console.print(f"[bold red]Context build failed:[/bold red] {e}")

@app.command()
def doctor(workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace scope.")):
    """Deep diagnostics for memory indexing & storage with production metrics."""
    workspace_id = workspace or get_workspace_id()
    try:
        if is_running():
            diag = requests.get(
                f"{URL}/doctor",
                params={"workspace_id": workspace_id},
                headers=_daemon_headers(),
            ).json()
        else:
            service = get_service()
            diag = service.get_diagnostics(workspace_id)

        stats = diag["db_stats"]
        states = diag["memory_health"]

        console.print("[bold blue]🩺 MemoryKernel Production Status[/bold blue]")
        console.print()
        
        # Database Section
        console.print("[bold]💾 Database[/bold]")
        console.print(f"  Schema Version: [cyan]{stats.get('schema_version', 'unknown')}[/cyan]")
        console.print(f"  Profile: [cyan]{stats.get('performance_profile', 'lite')}[/cyan] ({stats.get('index_mode', 'sqlite')})")
        console.print(f"  FTS: [{'green' if stats.get('fts_available', False) else 'yellow'}]{'available' if stats.get('fts_available', False) else 'fallback'}[/]")
        console.print(f"  Journal Mode: [cyan]{stats.get('journal_mode', 'unknown').upper()}[/cyan]")
        console.print(f"  Database Size: [cyan]{stats.get('database_size_mb', 0):.2f} MB[/cyan]")
        if stats.get('wal_size_mb', 0) > 0:
            console.print(f"  WAL Size: [cyan]{stats.get('wal_size_mb', 0):.2f} MB[/cyan]")
        console.print()
        
        # Memory Section
        console.print("[bold]📦 Memory[/bold]")
        console.print(f"  Total Memories: [cyan]{stats['total_memories']}[/cyan]")
        console.print(f"  Total Facts: [cyan]{stats['total_active_facts']}[/cyan]")
        total_embedded = stats.get('embedded_memories', 0) + stats.get('embedded_facts', 0)
        total_items = stats['total_memories'] + stats['total_active_facts']
        embed_pct = (total_embedded / total_items * 100) if total_items > 0 else 0
        console.print(f"  Embedded: [cyan]{total_embedded} / {total_items}[/cyan] ([green]{embed_pct:.1f}%[/green])")
        
        if "runtime" in diag:
            runtime = diag["runtime"]
            console.print(f"  Index Size: [cyan]{runtime.get('index_entries', 0)} entries[/cyan]")
        console.print()
        
        # Memory Health
        console.print("[bold]🌡  Memory Health[/bold]")
        console.print(f"  🔥 Hot:  [bold red]{states['hot']}[/bold red]")
        console.print(f"  🌤  Warm: [bold yellow]{states['warm']}[/bold yellow]")
        console.print(f"  ❄  Cold: [bold blue]{states['cold']}[/bold blue]")
        console.print()

        # Performance Section
        if "runtime" in diag:
            cache = diag["runtime"].get("cache", {})
            console.print("[bold]⚡ Cache Performance[/bold]")
            for layer, s in cache.items():
                console.print(f"  [dim]•[/dim] {layer.capitalize():<11}: Hit Rate: [green]{s['hit_rate']}[/green] ({s['size']}/{s['max_size']})")
            console.print()
        
        # Background Jobs Section
        if "runtime" in diag:
            active_jobs = diag["runtime"].get("active_jobs", 0)
            console.print("[bold]🔧 Background Jobs[/bold]")
            console.print(f"  Active: [cyan]{active_jobs}[/cyan]")
            console.print()
        
        # Health Status
        health = "✓ HEALTHY"
        health_color = "green"
        
        # Check for issues
        if embed_pct < 90:
            health = "⚠ DEGRADED (Low embedding coverage)"
            health_color = "yellow"
        if stats.get('wal_size_mb', 0) > 100:
            health = "⚠ WARNING (Large WAL file)"
            health_color = "yellow"
        
        console.print(f"[bold]Health:[/bold] [{health_color}]{health}[/{health_color}]")
        
    except Exception as e:
        console.print(f"[bold red]Doctor failed:[/bold red] {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        console.print(f"[bold red]Doctor failed:[/bold red] {e}")

@app.command()
def jobs(
    watch: bool = False,
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace scope.")
):
    """List or watch background job status."""
    workspace_id = workspace or get_workspace_id()
    try:
        if not is_running():
            console.print("[yellow]Daemon not running. Direct service jobs not trackable yet.[/yellow]")
            return

        while True:
            resp = requests.get(
                f"{URL}/jobs",
                params={"workspace_id": workspace_id},
                headers=_daemon_headers(),
            ).json()
            table = Table(title="Background Jobs")
            table.add_column("ID", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Status", style="bold")
            table.add_column("Progress", justify="right")

            for job in resp["jobs"]:
                status_color = "green" if job["status"] == "completed" else "yellow"
                if job["status"] == "failed": status_color = "red"
                table.add_row(job["id"], job["type"], f"[{status_color}]{job['status']}[/{status_color}]", job["progress"])
            
            console.clear()
            console.print(table)
            if not watch: break
            time.sleep(1)
    except Exception as e:
        console.print(f"[bold red]Failed to fetch jobs:[/bold red] {e}")

@app.command()
def bench(iterations: int = 50, profile: bool = True):
    """Run latency benchmarks and identifying bottlenecks."""
    try:
        from memk.eval.benchmark import run_benchmarks, profile_breakdown
        
        console.print(f"[bold blue]🚀 Running MemoryKernel Latency Suite ({iterations} iterations)...[/bold blue]")
        
        if is_running():
            console.print("[dim]Daemon detected. Benchmarking HTTP Transport + RAM Index...[/dim]")
            run_benchmarks(service_mode=True)
        else:
            console.print("[dim]No daemon. Benchmarking Direct Service Layer...[/dim]")
            run_benchmarks(service_mode=False)

        if profile:
            profile_breakdown()
            
    except Exception as e:
        console.print(f"[bold red]Benchmark failed:[/bold red] {e}")

@app.command("synthesize-all")
def synthesize_all():
    """Build a complete knowledge base."""
    try:
        if is_running():
            resp = requests.post(f"{URL}/jobs/synthesize", headers=_daemon_headers()).json()
            console.print(f"[green]Job submitted![/green] Job ID: [cyan]{resp['job_id']}[/cyan]")
            return

        service = get_service()
        from memk.synthesis.synthesizer import KnowledgeSynthesizer
        runtime = service._get_runtime(get_workspace_id())
        files = KnowledgeSynthesizer(runtime.db).synthesize_all()
        console.print(f"[green]Completed![/green] Synthesized {len(files)} topic(s).")
    except Exception as e:
        console.print(f"[bold red]Global synthesis failed:[/bold red] {e}")

# --- Ingestion Commands ---

@app.command("ingest")
def ingest_git(
    limit: int = typer.Option(50, "--limit", "-n", help="Number of recent commits to ingest."),
    since: Optional[str] = typer.Option(None, "--since", help="Only commits after this date (e.g., '2024-01-01')."),
    branch: str = typer.Option("HEAD", "--branch", "-b", help="Git branch to ingest from."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without adding to memory."),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace scope.")
):
    """Ingest knowledge from Git commit history (metadata-first)."""
    workspace_id = workspace or get_workspace_id()
    
    try:
        from memk.ingestion.git_ingestor import GitIngestor
        from memk.workspace.manager import WorkspaceManager
        
        # Get repo path from workspace
        ws = WorkspaceManager()
        repo_path = ws.root
        
        console.print(f"[bold blue]📚 Ingesting Git History[/bold blue]")
        console.print(f"  Repository: [cyan]{repo_path}[/cyan]")
        console.print(f"  Branch: [cyan]{branch}[/cyan]")
        console.print(f"  Limit: [cyan]{limit}[/cyan] commits")
        if since:
            console.print(f"  Since: [cyan]{since}[/cyan]")
        
        # Create ingestor
        ingestor = GitIngestor(repo_path=str(repo_path))
        
        # Ingest commits
        with console.status("[bold green]Processing commits..."):
            memories = ingestor.ingest_commits(limit=limit, since=since, branch=branch)
        
        if not memories:
            console.print("[yellow]No commits matched ingestion rules.[/yellow]")
            return
        
        # Display summary
        console.print(f"\n[green]✓ Found {len(memories)} memory candidates[/green]")
        
        # Show preview
        if dry_run or len(memories) <= 10:
            table = Table(title="Memory Candidates")
            table.add_column("Category", style="magenta")
            table.add_column("Content", style="cyan")
            table.add_column("Importance", justify="right")
            
            for mem in memories[:10]:
                table.add_row(
                    mem["metadata"]["category"],
                    mem["content"][:60] + "..." if len(mem["content"]) > 60 else mem["content"],
                    f"{mem['importance']:.1f}"
                )
            
            console.print(table)
            
            if len(memories) > 10:
                console.print(f"[dim]... and {len(memories) - 10} more[/dim]")
        
        if dry_run:
            console.print("\n[yellow]Dry run - no memories added.[/yellow]")
            return
        
        # Add to memory
        console.print(f"\n[bold]Adding {len(memories)} memories to brain...[/bold]")
        
        service = get_service()
        import asyncio
        
        added_count = 0
        for mem in memories:
            try:
                asyncio.run(service.add_memory(
                    mem["content"],
                    importance=mem["importance"],
                    workspace_id=workspace_id
                ))
                added_count += 1
            except Exception as e:
                logger.warning(f"Failed to add memory: {e}")
        
        console.print(f"[green]✓ Successfully added {added_count}/{len(memories)} memories![/green]")
        
        # Show categories breakdown
        categories = {}
        for mem in memories:
            cat = mem["metadata"]["category"]
            categories[cat] = categories.get(cat, 0) + 1
        
        console.print("\n[bold]Categories:[/bold]")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            console.print(f"  • {cat}: [cyan]{count}[/cyan]")
        
    except Exception as e:
        console.print(f"[bold red]Ingestion failed:[/bold red] {e}")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(code=1)

# --- Watcher Commands ---

watch_app = typer.Typer(help="File watcher commands for real-time change detection")
app.add_typer(watch_app, name="watch")

@watch_app.command("start")
def watch_start(
    foreground: bool = typer.Option(False, "--foreground", "-f", help="Run in foreground (blocking).")
):
    """Start the file watcher for the current workspace."""
    try:
        from memk.workspace.manager import WorkspaceManager
        from memk.watcher.file_watcher import WatcherService
        
        ws = WorkspaceManager()
        if not ws.is_initialized():
            console.print("[red]Workspace not initialized. Run 'memk init' first.[/red]")
            raise typer.Exit(code=1)
        
        console.print(f"[bold blue]👁  Starting File Watcher[/bold blue]")
        console.print(f"  Workspace: [cyan]{ws.root}[/cyan]")
        
        watcher_service = WatcherService(str(ws.root), ws)
        watcher_service.start()
        
        console.print("[green]✓ File watcher started successfully![/green]")
        console.print("[dim]Monitoring workspace for changes...[/dim]")
        
        if foreground:
            console.print("\n[yellow]Press Ctrl+C to stop[/yellow]\n")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopping watcher...[/yellow]")
                watcher_service.stop()
                console.print("[green]Watcher stopped.[/green]")
        else:
            console.print("[yellow]Note: Watcher is running in this process. Use daemon mode for persistent watching.[/yellow]")
            console.print("[dim]Tip: Run 'memk serve' to start daemon with integrated watcher.[/dim]")
            
    except Exception as e:
        console.print(f"[bold red]Failed to start watcher:[/bold red] {e}")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(code=1)

@watch_app.command("stop")
def watch_stop():
    """Stop the file watcher (daemon mode only)."""
    try:
        if not is_running():
            console.print("[yellow]Daemon not running. No watcher to stop.[/yellow]")
            return
        
        # Send stop command to daemon
        resp = requests.post(f"{URL}/watcher/stop", headers=_daemon_headers()).json()
        
        if resp.get("success"):
            console.print("[green]✓ File watcher stopped.[/green]")
        else:
            console.print(f"[yellow]Watcher stop failed: {resp.get('message')}[/yellow]")
            
    except Exception as e:
        console.print(f"[bold red]Failed to stop watcher:[/bold red] {e}")

@watch_app.command("status")
def watch_status():
    """Show file watcher status and statistics."""
    try:
        if is_running():
            # Get status from daemon
            resp = requests.get(f"{URL}/watcher/status", headers=_daemon_headers()).json()
            status = resp.get("status", {})
        else:
            console.print("[yellow]Daemon not running. Checking local workspace...[/yellow]")
            status = {"running": False}
        
        console.print("[bold blue]👁  File Watcher Status[/bold blue]")
        
        running = status.get("running", False)
        color = "green" if running else "red"
        console.print(f"  Status: [{color}]{'RUNNING' if running else 'STOPPED'}[/{color}]")
        
        if running:
            console.print(f"  Workspace: [cyan]{status.get('workspace_root', 'N/A')}[/cyan]")
            console.print(f"  Uptime: [cyan]{status.get('uptime_seconds', 0):.0f}s[/cyan]")
            console.print(f"\n[bold]Statistics:[/bold]")
            console.print(f"  Total Events: [cyan]{status.get('total_events', 0)}[/cyan]")
            console.print(f"  Filtered: [dim]{status.get('filtered_events', 0)}[/dim]")
            console.print(f"  Batched: [cyan]{status.get('batched_events', 0)}[/cyan]")
            console.print(f"  Generation Bumps: [yellow]{status.get('generation_bumps', 0)}[/yellow]")
            console.print(f"  Pending: [dim]{status.get('pending_events', 0)}[/dim]")
            
            # Show recent changes
            recent = status.get("recent_changes", [])
            if recent:
                console.print(f"\n[bold]Recent Changes:[/bold]")
                for change in recent[-5:]:
                    console.print(f"  [dim]{change['timestamp']}[/dim] → Gen {change['generation']} ({change['event_count']} files)")
        
    except Exception as e:
        console.print(f"[bold red]Failed to get watcher status:[/bold red] {e}")

# --- Sync Commands ---

sync_app = typer.Typer(help="Synchronization management and observability")
app.add_typer(sync_app, name="sync")

@sync_app.command("stats")
def sync_stats(workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace scope.")):
    """Health metrics for Delta Sync and Merkle Tree hardening."""
    workspace_id = workspace or get_workspace_id()
    try:
        if is_running():
            # For simplicity, we fallback to local service if daemon doesn't have the endpoint yet
            try:
                resp = requests.get(
                    f"{URL}/sync/stats",
                    params={"workspace_id": workspace_id},
                    headers=_daemon_headers(),
                ).json()
                if "error" not in resp:
                    stats = resp
                else: raise Exception(resp["error"])
            except:
                service = get_service()
                from memk.workspace.manager import WorkspaceManager
                ws = WorkspaceManager()
                runtime = service.global_runtime.get_workspace_runtime(workspace_id, ws.get_db_path())
                from memk.sync.stats import SyncStatsService
                stats = SyncStatsService(runtime).get_sync_hardening_stats()
        else:
            service = get_service()
            from memk.workspace.manager import WorkspaceManager
            ws = WorkspaceManager()
            runtime = service.global_runtime.get_workspace_runtime(workspace_id, ws.get_db_path())
            from memk.sync.stats import SyncStatsService
            stats = SyncStatsService(runtime).get_sync_hardening_stats()

        if "error" in stats:
            console.print(f"[bold red]Sync stats error:[/bold red] {stats['error']}")
            return

        console.print("[bold blue]🔄 Delta Sync Hardening Metrics[/bold blue]")
        console.print()
        
        # Oplog
        console.print("[bold]📜 Oplog (Write Log)[/bold]")
        console.print(f"  Count: [cyan]{stats['oplog']['count']}[/cyan]")
        console.print(f"  Oldest Entry: [cyan]{stats['oplog']['oldest_age_seconds']:.1f}s[/cyan] ago")
        console.print(f"  Prunable: [cyan]{stats['oplog']['prunable_count']}[/cyan] entries")
        console.print()
        
        # Replicas
        console.print("[bold]👯 Replicas[/bold]")
        console.print(f"  Active Replicas: [cyan]{stats['replicas']['checkpoint_count']}[/cyan]")
        console.print(f"  Slowest Lag: [yellow]{stats['replicas']['slowest_lag_seconds']:.1f}s[/yellow]")
        console.print()
        
        # Integrity
        console.print("[bold]🛡  Data Integrity[/bold]")
        stale_hash = stats['integrity']['stale_row_hash_count']
        stale_bucket = stats['integrity']['stale_merkle_bucket_count']
        console.print(f"  Stale Row Hashes: [{'red' if stale_hash > 0 else 'green'}]{stale_hash}[/]")
        console.print(f"  Stale Merkle Buckets: [{'red' if stale_bucket > 0 else 'green'}]{stale_bucket}[/]")
        console.print()
        
        # GC
        console.print("[bold]🧹 Garbage Collection[/bold]")
        console.print(f"  Last Run: [cyan]{stats['gc']['last_run']}[/cyan]")
        console.print(f"  Last Pruned: [cyan]{stats['gc']['last_deleted_count']}[/cyan] items")
        
    except Exception as e:
        console.print(f"[bold red]Sync stats failed:[/bold red] {e}")

if __name__ == "__main__":
    app()
