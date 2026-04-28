"""
Research notebook memory example.

Run `memk serve` first. This example stores a few research notes and retrieves
context for a follow-up writing task.
"""

from memk.sdk import MemoryKernel


def main() -> None:
    mk = MemoryKernel()

    notes = [
        "Paper note: Hybrid retrieval improves recall when semantic vectors miss exact identifiers.",
        "Paper note: Graph propagation can recover multi-hop context across entities and claims.",
        "Experiment note: Deterministic offline embeddings are useful for repeatable CI benchmarks.",
    ]

    for note in notes:
        mk.remember(note, importance=0.7)

    print("Related notes:")
    for item in mk.search("multi-hop retrieval benchmark", limit=3):
        print(f"- {item.score:.3f}: {item.content}")

    print("\nContext:")
    print(mk.context("What should I mention about retrieval benchmarks?"))


if __name__ == "__main__":
    main()
