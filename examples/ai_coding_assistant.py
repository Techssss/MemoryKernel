"""
AI Coding Assistant with MemoryKernel
======================================
A practical example demonstrating MemoryKernel's capabilities for building
an AI coding assistant that remembers your coding style and project context.

Features:
- Learn from Git history
- Remember coding patterns
- Context-aware code suggestions
- Track architecture decisions
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from memk.core.runtime_v2 import get_runtime_v2
from memk.workspace.manager import WorkspaceManager


class AICodingAssistant:
    """
    AI Coding Assistant powered by MemoryKernel.
    
    This assistant:
    - Learns from your Git history
    - Remembers coding patterns and decisions
    - Provides context-aware suggestions
    - Tracks your preferences over time
    """
    
    def __init__(self, workspace_id: str = "coding-assistant"):
        self.workspace_id = workspace_id
        self.runtime_manager = get_runtime_v2()
        self.workspace = None
        
    async def initialize(self):
        """Initialize the assistant and load workspace."""
        print("🤖 Initializing AI Coding Assistant...")
        
        # Initialize workspace manager
        try:
            ws_manager = WorkspaceManager()
            if not ws_manager.is_initialized():
                print("   Initializing workspace...")
                ws_manager.initialize()
        except Exception as e:
            print(f"   Note: {e}")
        
        # Get workspace runtime
        self.workspace = self.runtime_manager.get_workspace_runtime(self.workspace_id)
        
        print(f"✓ Assistant ready! (Generation: {self.workspace.get_generation()})")
        print()
    
    async def learn_from_git(self, limit: int = 50):
        """Learn coding patterns from Git history."""
        print(f"📚 Learning from Git history (last {limit} commits)...")
        
        try:
            from memk.ingestion.git_ingestor import GitIngestor
            
            ingestor = GitIngestor(self.workspace.db)
            stats = ingestor.ingest_history(limit=limit)
            
            print(f"✓ Learned from {stats['commits_processed']} commits")
            print(f"  - {stats['memories_created']} memories created")
            print(f"  - {stats['facts_extracted']} facts extracted")
            
            # Bump generation after learning
            new_gen = self.workspace.bump_generation()
            print(f"  - Generation bumped to {new_gen}")
            print()
            
        except Exception as e:
            print(f"⚠ Could not learn from Git: {e}")
            print()
    
    async def remember(self, content: str, importance: float = 0.5):
        """Remember a coding pattern or decision."""
        print(f"💾 Remembering: {content[:60]}...")
        
        # Get embedder from container
        embedder = self.runtime_manager.container.get_embedder()
        
        # Embed content
        embedding = embedder.embed(content)
        
        # Store in database
        mem_id = self.workspace.db.insert_memory(
            content,
            embedding=embedding,
            importance=importance
        )
        
        # Add to index
        from memk.retrieval.index import IndexEntry
        import datetime
        
        entry = IndexEntry(
            id=mem_id,
            item_type="memory",
            content=content,
            importance=importance,
            confidence=1.0,
            created_at=datetime.datetime.now().isoformat(),
            decay_score=1.0,
            access_count=0,
        )
        self.workspace.index.add_entry(entry, embedding)
        
        # Bump generation
        new_gen = self.workspace.bump_generation()
        
        print(f"✓ Remembered (ID: {mem_id}, Generation: {new_gen})")
        print()
        
        return mem_id
    
    async def ask(self, question: str, limit: int = 5):
        """Ask the assistant a question about the codebase."""
        print(f"❓ Question: {question}")
        print()
        
        # Search for relevant context
        results = self.workspace.retriever.retrieve(question, limit=limit)
        
        if not results:
            print("💭 I don't have enough context to answer that yet.")
            print("   Try teaching me by:")
            print("   - Running learn_from_git()")
            print("   - Adding memories with remember()")
            print()
            return
        
        print(f"📖 Found {len(results)} relevant memories:")
        print()
        
        for i, item in enumerate(results, 1):
            print(f"{i}. [{item.item_type}] (score: {item.score:.3f})")
            print(f"   {item.content[:100]}...")
            if item.breakdown:
                print(f"   Breakdown: vec={item.breakdown.vector_similarity:.2f}, "
                      f"kw={item.breakdown.keyword_score:.2f}, "
                      f"imp={item.breakdown.importance:.2f}")
            print()
        
        # Build context
        context = self.workspace.builder.build_context(results)
        
        print("🧠 Context Summary:")
        print("-" * 60)
        print(context)
        print("-" * 60)
        print()
        
        return results
    
    async def suggest_code(self, description: str):
        """Suggest code based on learned patterns."""
        print(f"💡 Suggesting code for: {description}")
        print()
        
        # Search for similar patterns
        results = self.workspace.retriever.retrieve(description, limit=3)
        
        if not results:
            print("💭 No similar patterns found. Try learning from Git first.")
            print()
            return
        
        print("📝 Similar patterns found:")
        print()
        
        for i, item in enumerate(results, 1):
            print(f"{i}. {item.content[:80]}...")
            print(f"   Score: {item.score:.3f}, Importance: {item.importance:.2f}")
            print()
        
        print("💡 Suggestion: Based on these patterns, you might want to:")
        print("   - Follow similar structure")
        print("   - Use consistent naming conventions")
        print("   - Apply the same design patterns")
        print()
    
    async def track_decision(self, decision: str, reason: str):
        """Track an architecture or design decision."""
        print(f"📋 Tracking decision: {decision}")
        
        content = f"DECISION: {decision}\nREASON: {reason}"
        
        # Store with high importance
        await self.remember(content, importance=0.9)
        
        print(f"✓ Decision tracked with high importance")
        print()
    
    async def get_stats(self):
        """Get assistant statistics."""
        print("📊 Assistant Statistics:")
        print()
        
        stats = self.workspace.db.get_stats()
        diag = self.workspace.get_diagnostics()
        
        print(f"Memories: {stats['total_memories']}")
        print(f"Facts: {stats['total_active_facts']}")
        print(f"Index entries: {diag['index_entries']}")
        print(f"Generation: {diag['generation']}")
        print(f"Cache hit rate: {diag['cache']['query']['hit_rate']:.2%}")
        print()


async def demo():
    """Run a complete demo of the AI Coding Assistant."""
    print("=" * 70)
    print("AI Coding Assistant Demo")
    print("=" * 70)
    print()
    
    # Initialize assistant
    assistant = AICodingAssistant()
    await assistant.initialize()
    
    # Demo 1: Learn from Git
    print("DEMO 1: Learning from Git History")
    print("-" * 70)
    await assistant.learn_from_git(limit=20)
    
    # Demo 2: Remember coding patterns
    print("DEMO 2: Remembering Coding Patterns")
    print("-" * 70)
    await assistant.remember(
        "Always use dependency injection for better testability",
        importance=0.8
    )
    await assistant.remember(
        "Prefer composition over inheritance for flexibility",
        importance=0.7
    )
    await assistant.remember(
        "Use protocol-based interfaces instead of abstract base classes",
        importance=0.9
    )
    
    # Demo 3: Ask questions
    print("DEMO 3: Asking Questions")
    print("-" * 70)
    await assistant.ask("How should I structure my code for testability?")
    
    # Demo 4: Suggest code
    print("DEMO 4: Code Suggestions")
    print("-" * 70)
    await assistant.suggest_code("Create a new service class")
    
    # Demo 5: Track decisions
    print("DEMO 5: Tracking Architecture Decisions")
    print("-" * 70)
    await assistant.track_decision(
        "Use SQLite with WAL mode for storage",
        "Provides good performance with ACID guarantees and no external dependencies"
    )
    
    # Demo 6: Get statistics
    print("DEMO 6: Statistics")
    print("-" * 70)
    await assistant.get_stats()
    
    print("=" * 70)
    print("✅ Demo Complete!")
    print("=" * 70)


async def interactive_mode():
    """Run the assistant in interactive mode."""
    print("=" * 70)
    print("AI Coding Assistant - Interactive Mode")
    print("=" * 70)
    print()
    
    assistant = AICodingAssistant()
    await assistant.initialize()
    
    print("Commands:")
    print("  learn [N]        - Learn from last N Git commits")
    print("  remember <text>  - Remember a coding pattern")
    print("  ask <question>   - Ask a question")
    print("  suggest <desc>   - Get code suggestions")
    print("  decide <text>    - Track a decision")
    print("  stats            - Show statistics")
    print("  help             - Show this help")
    print("  exit             - Exit")
    print()
    
    while True:
        try:
            command = input("🤖 > ").strip()
            
            if not command:
                continue
            
            parts = command.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if cmd == "exit":
                print("Goodbye! 👋")
                break
            
            elif cmd == "help":
                print("Commands: learn, remember, ask, suggest, decide, stats, exit")
            
            elif cmd == "learn":
                limit = int(args) if args else 20
                await assistant.learn_from_git(limit)
            
            elif cmd == "remember":
                if not args:
                    print("Usage: remember <text>")
                else:
                    await assistant.remember(args)
            
            elif cmd == "ask":
                if not args:
                    print("Usage: ask <question>")
                else:
                    await assistant.ask(args)
            
            elif cmd == "suggest":
                if not args:
                    print("Usage: suggest <description>")
                else:
                    await assistant.suggest_code(args)
            
            elif cmd == "decide":
                if not args:
                    print("Usage: decide <decision>")
                else:
                    reason = input("Reason: ").strip()
                    await assistant.track_decision(args, reason)
            
            elif cmd == "stats":
                await assistant.get_stats()
            
            else:
                print(f"Unknown command: {cmd}")
                print("Type 'help' for available commands")
            
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")
            print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(interactive_mode())
    else:
        asyncio.run(demo())
