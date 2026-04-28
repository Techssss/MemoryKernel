/**
 * @memk/sdk
 * =========
 * MemoryKernel SDK for Node.js
 * 
 * Simple, type-safe client for integrating local project memory.
 */

export { MemoryKernel, MemoryKernel as MemoryKernelClient } from './client';
export type { MemoryKernelOptions } from './client';
export type {
  MemoryItem,
  WorkspaceStatus,
  RememberOptions,
  SearchOptions,
  ContextOptions,
  IngestGitOptions,
  IngestGitResult
} from './types';
