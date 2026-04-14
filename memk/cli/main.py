import typer
import requests
import time
import datetime
from rich.console import Console
from rich.table import Table
from memk.server.manager import URL, is_running
from memk.core.service import MemoryKernelService

app = typer.Typer(
    help="MemoryKernel (memk) - Local-first memory infrastructure for AI agents",
    no_args_is_help=True
)
console = Console()
_service_instance = None

def get_service() -> MemoryKernelService:
    global _service_instance
    if _service_instance is None:
        _service_instance = MemoryKernelService()
    return _service_instance

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
    """Check the status of the local daemon."""
    from memk.server import manager
    stat = manager.get_status()
    color = "green" if "RUNNING" in stat else "red"
    console.print(f"Daemon Status: [{color}]{stat}[/{color}]")
    console.print(f"Endpoint: [cyan]{manager.URL}[/cyan]")

# --- Core Commands ---

@app.command()
def init():
    """Initialize the MemoryKernel storage."""
    try:
        service = get_service()
        service.ensure_initialized()
        service.runtime.db.init_db()
        console.print("[green]MemoryKernel database initialized successfully.[/green]")
    except Exception as e:
        console.print(f"[bold red]Failed to initialize database:[/bold red] {e}")
        raise typer.Exit(code=1)

@app.command()
def add(
    content: str,
    importance: float = typer.Option(0.5, "--importance", "-i", min=0, max=1, help="Priority of this memory."),
    confidence: float = typer.Option(1.0, "--confidence", "-c", min=0, max=1, help="Certainty of this memory."),
    workspace: str = typer.Option("default", "--workspace", "-w", help="Workspace scope.")
):
    """Add a new memory fact (Write-Time Embedding)."""
    try:
        if is_running():
            resp = requests.post(f"{URL}/add", json={
                "content": content, "importance": importance, "confidence": confidence, "workspace_id": workspace
            }).json()
            console.print(f"[green]Added via Daemon![/green] ID: [cyan]{resp['id'][:8]}...[/cyan]")
            return

        service = get_service()
        res = service.add_memory(content, importance, confidence, workspace)
        console.print(f"[green]Added memory successfully![/green] ID: [cyan]{res['id'][:8]}...[/cyan]")
    except Exception as e:
        console.print(f"[bold red]Failed to add memory:[/bold red] {e}")
        raise typer.Exit(code=1)

@app.command()
def search(
    query: str, 
    limit: int = typer.Option(10, "--limit", "-l", help="Number of results."),
    workspace: str = typer.Option("default", "--workspace", "-w", help="Workspace scope.")
):
    """Retrieve memories using in-memory vector search."""
    try:
        if is_running():
            resp = requests.post(f"{URL}/search", json={"query": query, "limit": limit, "workspace_id": workspace}).json()
            results = resp["results"]
        else:
            service = get_service()
            results = service.search(query, limit, workspace)

        table = Table(title=f"Search Results for: '{query}'")
        table.add_column("Type", style="dim")
        table.add_column("Content", style="cyan")
        table.add_column("Score", justify="right")

        for r in results:
            table.add_row(r["item_type"], r["content"], f"{r['score']:.3f}")
        
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Search failed:[/bold red] {e}")

@app.command()
def build_context(
    query: str, 
    max_chars: int = typer.Option(500, help="Maximum characters for token budget."),
    threshold: float = typer.Option(0.3, help="Minimum score to include."),
    workspace: str = typer.Option("default", "--workspace", "-w", help="Workspace scope.")
):
    """Build compact, LLM-ready context (Fast retrieval)."""
    try:
        if is_running():
            resp = requests.post(f"{URL}/context", json={
                "query": query, "max_chars": max_chars, "threshold": threshold, "workspace_id": workspace
            }).json()
            context = resp["context"]
        else:
            service = get_service()
            context = service.build_context(query, max_chars, threshold, workspace)

        console.print("\n[bold]Generated Context:[/bold]")
        console.print(f"[dim]{'-' * 40}[/dim]")
        console.print(context)
        console.print(f"[dim]{'-' * 40}[/dim]")
    except Exception as e:
        console.print(f"[bold red]Context build failed:[/bold red] {e}")

@app.command()
def doctor():
    """Run diagnostics on the memory kernel."""
    try:
        if is_running():
            data = requests.get(f"{URL}/doctor").json()
        else:
            service = get_service()
            data = service.get_diagnostics()

        stats = data["db_stats"]
        states = data["memory_health"]

        console.print("[bold blue]🩺 MemoryKernel Doctor Report[/bold blue]")
        console.print(f"📦 Total Memories: [cyan]{stats['total_memories']}[/cyan]")
        console.print(f"🧠 Total Facts:    [cyan]{stats['total_active_facts']}[/cyan]")
        console.print(f"\n[bold]🌡  Memory Health[/bold]")
        console.print(f"  🔥 Hot:  [bold red]{states['hot']}[/bold red]")
        console.print(f"  🌤  Warm: [bold yellow]{states['warm']}[/bold yellow]")
        console.print(f"  ❄  Cold: [bold blue]{states['cold']}[/bold blue]")

        if "runtime" in data:
            cache = data["runtime"].get("cache", {})
            console.print(f"\n[bold]⚡ Cache Performance[/bold]")
            for layer, s in cache.items():
                console.print(f"  [dim]•[/dim] {layer.capitalize():<11}: Hit Rate: [green]{s['hit_rate']}[/green] ({s['size']}/{s['max_size']})")
    except Exception as e:
        console.print(f"[bold red]Doctor failed:[/bold red] {e}")

@app.command()
def jobs(watch: bool = False):
    """List or watch background job status."""
    try:
        if not is_running():
            console.print("[yellow]Daemon not running. Direct service jobs not trackable yet.[/yellow]")
            return

        while True:
            resp = requests.get(f"{URL}/jobs").json()
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
            resp = requests.post(f"{URL}/jobs/synthesize").json()
            console.print(f"[green]Job submitted![/green] Job ID: [cyan]{resp['job_id']}[/cyan]")
            return

        service = get_service()
        from memk.synthesis.synthesizer import KnowledgeSynthesizer
        files = KnowledgeSynthesizer(service.runtime.db).synthesize_all()
        console.print(f"[green]Completed![/green] Synthesized {len(files)} topic(s).")
    except Exception as e:
        console.print(f"[bold red]Global synthesis failed:[/bold red] {e}")

if __name__ == "__main__":
    app()
