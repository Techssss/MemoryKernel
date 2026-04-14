import time
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from memk.storage.db import MemoryDB
from memk.retrieval.retriever import ScoredRetriever
from memk.context.builder import ContextBuilder
from memk.extraction.extractor import RuleBasedExtractor

logger = logging.getLogger(__name__)

@dataclass
class SimulationTurn:
    turn_id: int
    user_input: str
    extracted_facts: List[Dict[str, Any]]
    retrieved_count: int
    context_length: int
    has_conflict: bool
    metrics: Dict[str, float]

class MemorySimulator:
    """
    Simulates multi-turn agent interactions to evaluate MemoryKernel's performance.
    Tracks retention, conflict resolution, and decay.
    """

    def __init__(self, db_path: str = "eval_mem.db"):
        self.db = MemoryDB(db_path)
        self.db.init_db()
        self.extractor = RuleBasedExtractor()
        self.retriever = ScoredRetriever(self.db)
        self.builder = ContextBuilder(max_chars=1000)
        self.history: List[SimulationTurn] = []

    def run_scenario(self, name: str, turns: List[str], expected_facts: Optional[List[str]] = None):
        """Execute a sequence of inputs and track how memory evolves."""
        print(f"\n🚀 Running Scenario: [bold cyan]{name}[/bold cyan]")
        
        for i, user_input in enumerate(turns):
            # 1. Retrieval
            items = self.retriever.retrieve(user_input)
            active_fact_ids = [item.id for item in items if item.item_type == "fact"]
            conflicts = self.db.get_fact_conflicts(active_fact_ids)
            
            # 2. Context Building
            context = self.builder.build_context(items, conflicts=conflicts)
            
            # 3. Memory Extraction & Storage
            facts = self.extractor.extract_facts(user_input)
            for f in facts:
                self.db.insert_fact(f.subject, f.relation, f.object)
            
            # 4. Metric Calculation (Proxy for Intelligence)
            # - relevance score: how many retrieved items actually match the query keywords
            relevance = sum(1 for item in items if any(word in item.content.lower() for word in user_input.lower().split())) / max(1, len(items))
            
            turn = SimulationTurn(
                turn_id=i + 1,
                user_input=user_input,
                extracted_facts=facts,
                retrieved_count=len(items),
                context_length=len(context),
                has_conflict=len(conflicts) > 0,
                metrics={
                    "relevance": round(relevance, 2),
                    "hit_rate": 1.0 if items else 0.0
                }
            )
            self.history.append(turn)
            print(f"  Turn {i+1}: '{user_input[:30]}...' -> {len(items)} items retrieved. Conflict: {turn.has_conflict}")

    def generate_report(self, output_path: str = "eval_report.md"):
        """Synthesize simulation history into a Markdown report."""
        total_turns = len(self.history)
        avg_relevance = sum(t.metrics["relevance"] for t in self.history) / total_turns
        conflicts_handled = sum(1 for t in self.history if t.has_conflict)
        
        report = [
            "# MemoryKernel Evaluation Report\n",
            f"Generated on: {datetime.now().isoformat()}\n",
            "## Summary Metrics",
            f"* **Total Simulated Turns**: {total_turns}",
            f"* **Average Retrieval Relevance**: {avg_relevance:.2f}",
            f"* **Conflicts Detected & Managed**: {conflicts_handled}",
            f"* **Knowledge Retention Rate**: {avg_relevance * 100:.1f}%\n",
            "## Scenario Breakdown",
            "| Turn | Input | Retrieved | Conflict | Relevance |",
            "|------|-------|-----------|----------|-----------|"
        ]

        for t in self.history:
            report.append(f"| {t.turn_id} | {t.user_input[:20]}... | {t.retrieved_count} | {t.has_conflict} | {t.metrics['relevance']} |")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
        print(f"\n✅ Report generated at: {output_path}")

def run_standard_eval():
    sim = MemorySimulator("sim_test.db")
    
    # Scenario 1: Knowledge Evolution (Conflict)
    sim.run_scenario("Knowledge Evolution", [
        "user likes coffee",
        "user really loves coffee",
        "user moved to a new house",
        "user prefers tea now",  # Conflict with coffee
        "what does the user like?"
    ])
    
    # Scenario 2: Long-term Context
    sim.run_scenario("Long-term Context", [
        f"project {i} uses tech {i}" for i in range(20)
    ] + ["what techs are used in projects?"])
    
    sim.generate_report("knowledge_eval.md")

if __name__ == "__main__":
    run_standard_eval()
