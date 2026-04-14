# MemoryKernel Performance Benchmarks 🚀

This project is a simple benchmark suite to measure the latency of core `MemoryKernel` operations.

## Measured Operations

1.  **Insertion**: Time taken to embed, store, and index a new memory.
2.  **Search (Cold)**: Time taken to perform a search query without caching.
3.  **Search (Hot)**: Time taken to retrieve results from the internal cache.
4.  **Context Building**: Time taken to retrieve, reconcile, and format a context string for LLM prompts.

## How to Run

1.  Ensure you have the dependencies installed:
    ```bash
    pip install rich
    ```
2.  Run the script:
    ```bash
    python benchmarks/run_bench.py
    ```

## Expected Performance Targets

- **Insertion**: < 50ms (dominated by embedding generation)
- **Search (Cold)**: < 10ms
- **Search (Hot)**: < 1ms
- **Context Building**: < 20ms
