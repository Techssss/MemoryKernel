import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from memk.retrieval.retriever import RetrievedItem

logger = logging.getLogger(__name__)

class ContextBuilder:
    """
    Transforms retrieved database items into a structured, LLM-optimized context payload.
    
    Sections:
    - [Recent Memories]  : Fresh logs (short-term)
    - [Stable Facts]    : High-certainty knowledge
    - [Conflicts]       : Historical vs Current status
    - [User Preferences]: Personalized settings
    - [Summary]         : Auto-generated overview
    """

    def __init__(self, max_chars: int = 2000, st_memory_days: float = 7.0):
        """
        Parameters
        ----------
        max_chars      : Strict character budget for the total context.
        st_memory_days : Threshold for 'Recent' memories (days).
        """
        self.max_chars = max_chars
        self.st_memory_days = st_memory_days

    def build_context(
        self, 
        items: List[RetrievedItem], 
        conflicts: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Organize items into prioritized sections and pack them until max_chars is reached.
        """
        if not items:
            return "No relevant memories found."

        # 1. Categorize items
        user_prefs = []
        stable_facts = []
        recent_mems = []
        
        now = datetime.now(tz=timezone.utc)
        
        for item in items:
            # Parse age for short-term vs long-term check
            try:
                dt = datetime.fromisoformat(item.created_at.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age_days = (now - dt).total_seconds() / 86400.0
            except:
                age_days = 999
            
            if item.item_type == "fact":
                subj = item.content.split(" ")[0].lower()
                is_user = any(x in subj for x in ["user", "tôi", "tao", "me"])
                
                if is_user:
                    user_prefs.append(item)
                else:
                    stable_facts.append(item)
            else:
                if age_days <= self.st_memory_days:
                    recent_mems.append(item)
                else:
                    # Long-term memories are lower priority, maybe skip or add to facts?
                    # For now, let's keep them in recent if they were retrieved (top-ranked)
                    recent_mems.append(item)

        # 2. Format Conflict section (if any)
        conflict_lines = []
        if conflicts:
            for c in conflicts:
                # Group by key would be better but simple list for now
                conflict_lines.append(f"! HISTORICAL: {c['subject']} {c['predicate']} was {c['object']} (replaced {c['created_at']})")

        # 3. Assemble sections with priority
        # Priority: User Preferences > Recent Memories > Stable Facts > Conflicts
        ordered_sections = [
            ("[User Preferences]", [f"• {i.content}" for i in user_prefs]),
            ("[Recent Memories]",  [f"→ {i.content}" for i in recent_mems]),
            ("[Stable Facts]",     [f"• {i.content}" for i in stable_facts]),
            ("[Conflicts]",        [f"⚠ {line}" for line in conflict_lines]),
        ]
        
        # 4. Packing Logic
        output_parts = []
        current_len = 0
        
        for title, lines in ordered_sections:
            if not lines: continue
            
            section_header = f"\n{title}"
            if current_len + len(section_header) + 5 > self.max_chars:
                break
                
            output_parts.append(section_header)
            current_len += len(section_header)
            
            for line in lines:
                if current_len + len(line) + 2 > self.max_chars:
                    output_parts.append("  [...truncated...]")
                    current_len += 20
                    break
                output_parts.append(line)
                current_len += len(line) + 1

        # 5. Generate Summary
        summary_text = self._generate_rule_based_summary(items)
        summary_section = f"\n[Summary]\n{summary_text}"
        
        if current_len + len(summary_section) <= self.max_chars:
            output_parts.append(summary_section)

        return "\n".join(output_parts).strip()

    def _generate_rule_based_summary(self, items: List[RetrievedItem]) -> str:
        """Simple heuristic summary of retrieved knowledge."""
        facts_count = sum(1 for i in items if i.item_type == "fact")
        mem_count = len(items) - facts_count
        
        subjects = set()
        for i in items:
            subjects.add(i.content.split(" ")[0].lower())
            
        subj_str = ", ".join(list(subjects)[:3])
        if len(subjects) > 3:
            subj_str += "..."
            
        return f"Retrieved {facts_count} facts and {mem_count} context logs related to: {subj_str}."
