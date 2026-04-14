from typing import List
from memk.retrieval.retriever import RetrievedItem

class ContextBuilder:
    """
    Transforms retrieved database items into a compact, prompt-ready context string.
    Respects a strict character budget to control LLM token usage.
    """
    def __init__(self, max_chars: int = 1500):
        self.max_chars = max_chars

    def build_context(self, retrieved_items: List[RetrievedItem]) -> str:
        """
        Group items, rank them by internal priority (User > Project > Memories),
        and pack them until the character budget is reached.
        """
        if not retrieved_items:
            return ""
            
        # Group items
        user_facts = []
        project_facts = []
        memories = []
        
        for item in retrieved_items:
            if item.item_type == "fact":
                # MVP subject detection based on first word
                subj = item.content.split(" ")[0].lower()
                if "user" in subj or "tôi" in subj or "tao" in subj:
                    user_facts.append(item.content)
                else:
                    project_facts.append(item.content)
            else:
                memories.append(item.content)
                
        # Build string iteratively respecting the maximum budget
        final_lines = []
        current_len = 0
        
        def try_add_section(title: str, lines: List[str]):
            nonlocal current_len
            if not lines:
                return
                
            title_str = f"{title}:"
            # Prevent starting a section if we don't even have space for the title
            if current_len + len(title_str) + 1 > self.max_chars:
                return
                
            final_lines.append(title_str)
            current_len += len(title_str) + 1
            
            for line in lines:
                line_str = f"  - {line}"
                # If adding this line exceeds budget, add truncation warning and stop section
                if current_len + len(line_str) + 1 > self.max_chars:
                    trunc_msg = "  - ...[TRUNCATED_DUE_TO_BUDGET]"
                    if current_len + len(trunc_msg) <= self.max_chars:
                        final_lines.append(trunc_msg)
                        current_len += len(trunc_msg) + 1
                    break
                final_lines.append(line_str)
                current_len += len(line_str) + 1
                
        # Strict Packing Priority: User Facts -> System/Project Facts -> Raw Noise (Memories)
        try_add_section("User facts", user_facts)
        try_add_section("Project facts", project_facts)
        try_add_section("Raw memories", memories)
        
        return "\n".join(final_lines)
