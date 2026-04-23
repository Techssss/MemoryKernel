import logging
from memk.core.runtime import WorkspaceRuntime

logger = logging.getLogger("memk.rejuvenation")

class MemoryRejuvenator:
    """
    Manages the lifecycle state machine for archived memory records.
    
    State Machine (using 'archived' DB field):
      [0] ACTIVE: Normal queries hit this, Consolidator scans this.
      [1] ARCHIVED: Hidden from main IVF/Brute-force vector searches. Subsumed by kg_fact.
      
    Transitions:
      ACTIVE -> ARCHIVED : Consolidator merged it into a fact.
      ARCHIVED -> ACTIVE (Rejuvenation) : 
         - Trigger 1: Graph Traversal "touches" the memory repeatedly (highly accessed).
         - Trigger 2: Explicit flag_for_reconsolidation (e.g. user update or logical contradiction with new fact).
    """

    def __init__(self, runtime: WorkspaceRuntime, access_threshold: int = 3):
        self.runtime = runtime
        self.db = runtime.db
        self.access_threshold = access_threshold
        
    def evaluate_memory_access(self, memory_id: str) -> bool:
        """
        Evaluate if an archived memory should be revived due to access threshold.
        Call this when an archived node is encountered via Graph Traversal or deep search.
        
        Returns True if the memory was successfully rejuvenated.
        """
        with self.db._get_connection() as conn:
            row = conn.execute("SELECT archived, access_count FROM memories WHERE id = ?", (memory_id,)).fetchone()
            
        if not row or row["archived"] != 1:
            return False 
            
        new_count = row["access_count"] + 1
        self.db.touch_memory(memory_id)
        
        if new_count >= self.access_threshold:
            self.db.unarchive_memory(memory_id)
            logger.info(f"[{self.runtime.workspace_id}] Rejuvenated memory {memory_id[:8]} (access > {self.access_threshold})")
            return True
            
        return False
        
    def flag_for_reconsolidation(self, memory_id: str, reason: str = "contradiction") -> bool:
        """
        Forces an archived memory to wake up. Often used when a conflicting fact is inserted.
        This throws the memory back into the ACTIVE pool so the Consolidator will reprocess it.
        """
        with self.db._get_connection() as conn:
            row = conn.execute("SELECT archived FROM memories WHERE id = ?", (memory_id,)).fetchone()
            
        if not row or row["archived"] == 0:
            return False
            
        self.db.unarchive_memory(memory_id)
        
        # Boost importance so it becomes a strong candidate in next consolidation
        try:
            with self.db._get_connection() as conn:
                conn.execute(
                    "UPDATE memories SET importance = MAX(importance, 0.8) WHERE id = ?", 
                    (memory_id,)
                )
        except Exception as e:
            logger.warning(f"Error boosting memory importance: {e}")
            
        logger.info(f"[{self.runtime.workspace_id}] Rejuvenated memory {memory_id[:8]} (Reason: {reason})")
        return True
