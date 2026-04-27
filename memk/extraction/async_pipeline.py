import logging
import time
from typing import Callable, Any

from memk.core.runtime import WorkspaceRuntime

logger = logging.getLogger("memk.extraction.async")

# Global GLiNER extractor singleton
_gliner_instance = None

def _get_gliner():
    global _gliner_instance
    if _gliner_instance is None:
        from memk.extraction.gliner_extractor import GLiNERAsyncExtractor
        _gliner_instance = GLiNERAsyncExtractor()
    return _gliner_instance

def enhanced_extraction_job(
    runtime: WorkspaceRuntime,
    workspace_id: str,
    memory_id: str,
    text: str,
    retries: int = 2,
    progress_callback: Callable = None,
    check_cancelled: Callable = None
) -> Any:
    """
    Background worker job for sophisticated extraction pipelines (e.g. GLiNER, LLM).
    Provides robust retry framing, cancellation checks, and progress reporting.
    """
    attempt = 0
    while attempt <= retries:
        if check_cancelled and check_cancelled():
            logger.info(f"[{workspace_id}] Enhanced extraction job cancelled for {memory_id[:8]}.")
            return

        attempt += 1
        logger.info(f"[{workspace_id}] Starting async extraction for {memory_id[:8]} (Attempt {attempt})")
        
        try:
            if progress_callback:
                progress_callback(0.2)
                
            # Perform advanced extraction via GLiNER
            extractor = _get_gliner()
            entities = extractor.extract_entities(text)
            repo = runtime.graph_repo
            
            if repo and entities:
                changed = False
                for ent in entities:
                    score = ent.get("score", 0.0)
                    if score >= 0.5:
                        e_id = repo.upsert_entity(workspace_id, ent["text"], confidence=score)
                        repo.add_mention(memory_id, e_id, role_hint=ent["label"])
                        changed = True
                        logger.debug(f"[{workspace_id}] Async Pipeline added GLiNER Entity: {ent['text']} ({ent['label']})")
                if changed:
                    runtime.refresh_graph_index()
            
            if progress_callback:
                progress_callback(1.0)
                
            logger.info(f"[{workspace_id}] ✓ Async extraction completed for {memory_id[:8]}")
            return {"status": "success", "attempt": attempt, "memory_id": memory_id}
            
        except Exception as e:
            logger.warning(f"[{workspace_id}] Async extraction exception on attempt {attempt}: {str(e)}")
            if attempt > retries:
                logger.error(f"[{workspace_id}] Max retries exhausted for memory {memory_id[:8]}. Fatal failure.")
                raise e
            
            # Exponential backoff before retry
            backoff_time = (2 ** attempt) * 0.5 
            time.sleep(backoff_time)
