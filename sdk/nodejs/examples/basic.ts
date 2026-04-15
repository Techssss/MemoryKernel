/**
 * Basic Node.js SDK Example
 * =========================
 * Demonstrates core MemoryKernel SDK operations.
 */

import { MemoryKernel } from '../src';

async function main() {
  // Initialize client (auto-detects workspace)
  const mk = new MemoryKernel();

  console.log('=== MemoryKernel SDK Demo ===\n');

  // 1. Remember some facts
  console.log('1. Adding memories...');
  await mk.remember("User prefers TypeScript over JavaScript", {
    importance: 0.8
  });
  await mk.remember("Project uses React for frontend", {
    importance: 0.7
  });
  await mk.remember("Database is PostgreSQL", {
    importance: 0.9
  });
  console.log('✓ Added 3 memories\n');

  // 2. Search for relevant information
  console.log("2. Searching for 'What language does user prefer?'...");
  const results = await mk.search("What language does user prefer?", {
    limit: 5
  });
  results.forEach(r => {
    console.log(`  [${r.score.toFixed(2)}] ${r.content}`);
  });
  console.log();

  // 3. Build context for LLM
  console.log('3. Building context...');
  const context = await mk.context("What should I know about this project?", {
    maxChars: 500
  });
  console.log(`Context (${context.length} chars):`);
  console.log(`  ${context.substring(0, 200)}...`);
  console.log();

  // 4. Check status
  console.log('4. Workspace status:');
  const status = await mk.status();
  console.log(`  Generation: ${status.generation}`);
  console.log(`  Memories: ${status.total_memories}`);
  console.log(`  Facts: ${status.total_facts}`);
  console.log(`  Watcher: ${status.watcher_running ? 'Running' : 'Stopped'}`);
  console.log();

  console.log('✓ Demo complete!');
}

main().catch(console.error);
