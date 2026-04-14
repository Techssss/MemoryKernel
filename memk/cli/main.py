import typer
from rich.console import Console
from rich.table import Table
from memk.storage.db import MemoryDB, DatabaseError
from memk.extraction.extractor import RuleBasedExtractor
from memk.retrieval.retriever import KeywordRetriever
from memk.context.builder import ContextBuilder

app = typer.Typer(
    help="MemoryKernel (memk) - Local-first memory infrastructure for AI agents",
    no_args_is_help=True
)
console = Console()
extractor = RuleBasedExtractor()

def get_db() -> MemoryDB:
    # MVP hardcodes db path, easily overridden later
    return MemoryDB()

@app.command()
def init():
    """Initialize the MemoryKernel storage."""
    try:
        db = get_db()
        db.init_db()
        console.print("[green]MemoryKernel database initialized successfully.[/green]")
    except DatabaseError as e:
        console.print(f"[bold red]Failed to initialize database:[/bold red] {e}")
        raise typer.Exit(code=1)

@app.command()
def add(content: str):
    """Add a new memory fact and automatically extract structured facts."""
    try:
        db = get_db()
        mem_id = db.insert_memory(content)
        console.print(f"[green]Added memory successfully![/green] ID: [cyan]{mem_id[:8]}...[/cyan]")
        
        facts = extractor.extract_facts(content)
        if facts:
            console.print(f"[blue]Found {len(facts)} structured fact(s). Storing...[/blue]")
            for f in facts:
                fact_id = db.insert_fact(
                    subject=f.subject, 
                    predicate=f.relation, 
                    obj=f.object
                )
                console.print(f"  - Extracted: ([bold]{f.subject}[/bold] -> [italic]{f.relation}[/italic] -> {f.object}) ID: [cyan]{fact_id[:8]}[/cyan]")

    except Exception as e:
        console.print(f"[bold red]Failed to add memory:[/bold red] {e}")
        raise typer.Exit(code=1)

@app.command()
def inspect(query: str):
    """Search and print matching memories by keyword."""
    try:
        db = get_db()
        results = db.search_memory(query)
        if not results:
            console.print(f"[yellow]No memories found for query:[/yellow] '{query}'")
            return
        
        table = Table(title=f"Memory Search Results for '{query}'")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Content", style="white")
        table.add_column("Created At", style="green")

        for row in results:
            table.add_row(str(row['id'])[:8] + '...', str(row['content']), str(row['created_at']))

        console.print(table)
    except DatabaseError as e:
        console.print(f"[bold red]Failed to search memories:[/bold red] {e}")
        raise typer.Exit(code=1)

@app.command("build-context")
def build_context(query: str, max_chars: int = typer.Option(500, help="Maximum characters for token budget.")):
    """Build compact, LLM-ready context from memory for a given query."""
    try:
        db = get_db()
        retriever = KeywordRetriever(db)
        builder = ContextBuilder(max_chars=max_chars)
        
        # 1. Retrieve items implicitly ranked and merged
        items = retriever.retrieve(query)
        if not items:
            console.print(f"[yellow]No context found for:[/yellow] '{query}'")
            return

        # 2. Compact them into strict token budget bounds
        context_str = builder.build_context(items)
        
        console.print(f"[bold blue]LLM Context Payload for:[/bold blue] '{query}' (Budget: {max_chars} chars)")
        console.print("--- BEGIN CONTEXT ---", style="dim")
        console.print(context_str, style="green")
        console.print("--- END CONTEXT ---", style="dim")
        
    except Exception as e:
        console.print(f"[bold red]Context process failed:[/bold red] {e}")
        raise typer.Exit(code=1)

@app.command()
def doctor():
    """Run diagnostics and observability checks on the MemoryKernel system."""
    try:
        db = get_db()
        stats = db.get_stats()
        console.print("[bold blue]🩺 MemoryKernel Doctor Report[/bold blue]\n")
        
        console.print(f"📦 Total Raw Memories: [cyan]{stats['total_memories']}[/cyan]")
        console.print(f"🧠 Total Active Facts: [cyan]{stats['total_active_facts']}[/cyan]")
        
        console.print("\n[bold]🕒 Recent Facts (Top 3)[/bold]")
        recent_facts = db.search_facts()[:3]
        for f in recent_facts:
            console.print(f"  - [green]({f['subject']} -> {f['predicate']} -> {f['object']})[/green] @ {f['created_at']}")
            
        console.print("\n[bold]🔍 Logical Conflict Check[/bold]")
        # Check if there's any identical subject+predicate combinations that somehow evaded reconciliation
        with db._get_connection() as conn:
            cur = conn.execute('''
                SELECT subject, predicate, COUNT(*) as count 
                FROM facts 
                WHERE is_active = 1 
                GROUP BY subject, predicate 
                HAVING count > 1
            ''')
            conflicts = cur.fetchall()
            if not conflicts:
                console.print("[green]✓ System is healthy. No duplicate active topics found.[/green]")
            else:
                console.print("[yellow]⚠ Detected active fact conflicts:[/yellow]")
                for row in conflicts:
                    console.print(f"  - Multiple active objects for: [bold]{row[0]} -> {row[1]}[/bold] (Count: {row[2]})")
                    
    except Exception as e:
        console.print(f"[bold red]Doctor failed:[/bold red] {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
