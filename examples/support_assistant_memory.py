"""
Support assistant memory example.

Run `memk serve` first. This example stores resolved incidents and retrieves
similar prior cases for a new ticket.
"""

from memk.sdk import MemoryKernel


def main() -> None:
    mk = MemoryKernel()

    cases = [
        "Resolved support case: SQLite permission errors were caused by a daemon running as another OS user.",
        "Resolved support case: Missing spaCy model caused extractor tests to skip, not fail.",
        "Resolved support case: MEMK_API_TOKEN mismatch caused 401 responses from protected daemon endpoints.",
    ]

    for case in cases:
        mk.remember(case, importance=0.8)

    print("Similar cases:")
    for item in mk.search("daemon returns 401 token", limit=3):
        print(f"- {item.score:.3f}: {item.content}")

    print("\nSuggested context:")
    print(mk.context("How do I troubleshoot daemon authentication errors?"))


if __name__ == "__main__":
    main()
