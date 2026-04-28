/**
 * MemoryKernel Client
 * ===================
 * Main SDK client for Node.js applications.
 */

import axios, { AxiosInstance } from 'axios';
import type {
  MemoryItem,
  WorkspaceStatus,
  RememberOptions,
  SearchOptions,
  ContextOptions,
  IngestGitOptions,
  IngestGitResult,
  APIResponse
} from './types';

const DEFAULT_DAEMON_URL = process.env.MEMK_DAEMON_URL || 'http://127.0.0.1:15301';

export interface MemoryKernelOptions {
  daemonUrl?: string;
  workspaceId?: string;
  apiToken?: string;
}

/**
 * MemoryKernel SDK Client
 * 
 * Simple interface for adding and retrieving memories from a local workspace.
 * 
 * @example
 * ```typescript
 * const mk = new MemoryKernel();
 * 
 * // Remember
 * await mk.remember("User prefers TypeScript");
 * 
 * // Search
 * const results = await mk.search("What does user prefer?");
 * results.forEach(r => {
 *   console.log(`${r.score.toFixed(2)}: ${r.content}`);
 * });
 * 
 * // Context
 * const context = await mk.context("What should I know?");
 * console.log(context);
 * ```
 */
export class MemoryKernel {
  private client: AxiosInstance;
  private workspaceId?: string;
  private _generation?: number;

  constructor(options: MemoryKernelOptions = {}) {
    const daemonUrl = options.daemonUrl || DEFAULT_DAEMON_URL;
    this.workspaceId = options.workspaceId;
    const apiToken = options.apiToken || process.env.MEMK_API_TOKEN;

    this.client = axios.create({
      baseURL: daemonUrl,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
        ...(apiToken ? { Authorization: `Bearer ${apiToken}` } : {})
      }
    });

    this.client.interceptors.response.use(
      response => response,
      (error: any) => {
        const detail = error?.response?.data?.detail;
        if (detail?.code) {
          error.message = `${detail.code}: ${detail.message || ''}`.trim();
        }
        return Promise.reject(error);
      }
    );

    // Check daemon is running
    this.checkDaemon().catch(() => {
      console.warn(
        `Daemon not running at ${daemonUrl}. Start with: memk serve`
      );
    });
  }

  /**
   * Check if daemon is running
   */
  private async checkDaemon(): Promise<void> {
    await this.client.get('/v1/health');
  }

  /**
   * Add a memory to the workspace
   * 
   * @param content - Memory content to store
   * @param options - Optional importance and confidence
   * @returns Memory ID
   * 
   * @example
   * ```typescript
   * const id = await mk.remember("User prefers dark mode", {
   *   importance: 0.8
   * });
   * ```
   */
  async remember(
    content: string,
    options: RememberOptions = {}
  ): Promise<string> {
    const payload = {
      content,
      importance: options.importance ?? 0.5,
      confidence: options.confidence ?? 1.0,
      workspace_id: this.workspaceId
    };

    const response = await this.client.post<APIResponse<{ id: string }>>('/v1/remember', payload);
    
    // Update generation
    this._generation = response.data.metadata.generation;

    return response.data.data.id;
  }

  /**
   * Search for relevant memories
   * 
   * @param query - Search query
   * @param options - Optional limit
   * @returns Array of memory items ranked by relevance
   * 
   * @example
   * ```typescript
   * const results = await mk.search("What does user prefer?", {
   *   limit: 5
   * });
   * ```
   */
  async search(
    query: string,
    options: SearchOptions = {}
  ): Promise<MemoryItem[]> {
    const payload = {
      query,
      limit: options.limit ?? 10,
      workspace_id: this.workspaceId,
      client_generation: this._generation
    };

    const response = await this.client.post<APIResponse<{ results: MemoryItem[] }>>(
      '/v1/search',
      payload
    );

    // Update generation
    this._generation = response.data.metadata.generation;

    // Check for staleness
    if (response.data.metadata.stale_warning) {
      console.warn(response.data.metadata.stale_warning);
    }

    return response.data.data.results;
  }

  /**
   * Build RAG context from relevant memories
   * 
   * @param query - Context query
   * @param options - Optional max_chars and threshold
   * @returns Formatted context string
   * 
   * @example
   * ```typescript
   * const context = await mk.context("What should I know?", {
   *   maxChars: 1000,
   *   threshold: 0.3
   * });
   * ```
   */
  async context(
    query: string,
    options: ContextOptions = {}
  ): Promise<string> {
    const payload = {
      query,
      max_chars: options.maxChars ?? 500,
      threshold: options.threshold ?? 0.3,
      workspace_id: this.workspaceId,
      client_generation: this._generation
    };

    const response = await this.client.post<APIResponse<{ context: string }>>(
      '/v1/context',
      payload
    );

    // Update generation
    this._generation = response.data.metadata.generation;

    // Check for staleness
    if (response.data.metadata.stale_warning) {
      console.warn(response.data.metadata.stale_warning);
    }

    return response.data.data.context;
  }

  /**
   * Get workspace status and statistics
   * 
   * @returns Workspace status object
   * 
   * @example
   * ```typescript
   * const status = await mk.status();
   * console.log(`Generation: ${status.generation}`);
   * console.log(`Memories: ${status.total_memories}`);
   * ```
   */
  async status(): Promise<WorkspaceStatus> {
    const params = this.workspaceId ? { workspace_id: this.workspaceId } : {};

    const response = await this.client.get<APIResponse<WorkspaceStatus>>(
      '/v1/status',
      { params }
    );

    const data = response.data.data;
    const stats = (data as any).stats || {};
    const watcher = (data as any).watcher || {};

    return {
      workspace_id: data.workspace_id,
      generation: data.generation,
      initialized: data.initialized,
      workspace_root: data.workspace_root,
      total_memories: stats.total_memories || 0,
      total_facts: stats.total_active_facts || 0,
      watcher_running: watcher.running || false
    };
  }

  /**
   * Ingest knowledge from Git commit history
   * 
   * @param options - Ingestion options (limit, since, branch)
   * @returns Ingestion result with count and categories
   * 
   * @example
   * ```typescript
   * const result = await mk.ingestGit({
   *   limit: 100,
   *   since: "2024-01-01"
   * });
   * console.log(`Ingested ${result.ingested_count} memories`);
   * ```
   */
  async ingestGit(options: IngestGitOptions = {}): Promise<IngestGitResult> {
    const payload = {
      limit: options.limit ?? 50,
      since: options.since,
      branch: options.branch ?? 'HEAD',
      workspace_id: this.workspaceId
    };

    const response = await this.client.post<APIResponse<IngestGitResult>>(
      '/v1/ingest/git',
      payload
    );

    // Update generation
    this._generation = response.data.metadata.generation;

    return response.data.data;
  }

  /**
   * Get current generation number
   */
  get generation(): number | undefined {
    return this._generation;
  }
}
