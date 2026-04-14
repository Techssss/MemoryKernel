from pydantic import BaseModel
from typing import List
from memk.storage.db import MemoryDB

class RetrievedItem(BaseModel):
    """Normalized payload representing either a structured fact or raw memory."""
    item_type: str  # "fact" or "memory"
    id: str
    content: str
    created_at: str
    score: float

class KeywordRetriever:
    """
    Retrieves facts and memories from the database using keyword matching.
    Designed as a separate layer to decouple business ranking logic from database queries.
    """
    def __init__(self, db: MemoryDB):
        self.db = db

    def retrieve(self, query: str, limit: int = 10) -> List[RetrievedItem]:
        query = query.strip()
        if not query:
            return []

        # 1. Fetch from facts
        raw_facts = self.db.search_facts(keyword=query)
        # 2. Fetch from raw memories
        raw_mems = self.db.search_memory(keyword=query)
        
        results: List[RetrievedItem] = []
        
        # Rank facts first (Higher conceptual density -> Base Score 2.0)
        for row in raw_facts:
            # Reconstruct a pseudo-sentence for the Context Builder to inject
            content_str = f"{row['subject']} {row['predicate']} {row['object']}"
            results.append(RetrievedItem(
                item_type="fact",
                id=row['id'],
                content=content_str,
                created_at=row['created_at'],
                score=2.0 
            ))
            
        # Rank memories lower (High noise threshold -> Base Score 1.0)
        for row in raw_mems:
            results.append(RetrievedItem(
                item_type="memory",
                id=row['id'],
                content=row['content'],
                created_at=row['created_at'],
                score=1.0
            ))
            
        # Sort results: First by score DESC (Facts > Memories), then by created_at DESC (Newer > Older)
        results.sort(key=lambda x: (x.score, x.created_at), reverse=True)
        
        return results[:limit]
