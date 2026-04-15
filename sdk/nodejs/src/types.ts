/**
 * Type definitions for MemoryKernel SDK
 */

export interface MemoryItem {
  item_type: string;
  id: string;
  content: string;
  score: number;
  importance: number;
  confidence: number;
  created_at: string;
  access_count?: number;
  decay_score?: number;
}

export interface WorkspaceStatus {
  workspace_id: string;
  generation: number;
  initialized: boolean;
  workspace_root: string;
  total_memories: number;
  total_facts: number;
  watcher_running: boolean;
}

export interface RememberOptions {
  importance?: number;
  confidence?: number;
}

export interface SearchOptions {
  limit?: number;
}

export interface ContextOptions {
  maxChars?: number;
  threshold?: number;
}

export interface IngestGitOptions {
  limit?: number;
  since?: string;
  branch?: string;
}

export interface IngestGitResult {
  ingested_count: number;
  categories: Record<string, number>;
}

export interface APIMetadata {
  workspace_id: string;
  generation: number;
  cache_hit: boolean;
  degraded: boolean;
  stale_warning?: string;
  timestamp: string;
}

export interface APIResponse<T> {
  data: T;
  metadata: APIMetadata;
}
