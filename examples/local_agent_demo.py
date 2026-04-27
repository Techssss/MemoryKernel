"""
Local agent demo with MemoryKernel.

This example keeps an async application loop, but the current Python SDK client is
synchronous. Calls into `MemoryKernel` are therefore plain method calls.
"""

import asyncio

from memk.sdk import MemoryKernel


class LocalAgent:
    """Simple local agent with persistent memory."""

    def __init__(self):
        self.client = MemoryKernel()

    async def remember_interaction(self, user_input: str, agent_response: str) -> None:
        """Store the interaction in local memory."""
        self.client.remember(
            f"User said: {user_input}",
            importance=0.7,
        )
        self.client.remember(
            f"Agent responded: {agent_response}",
            importance=0.5,
        )

    async def get_relevant_context(self, query: str, max_chars: int = 2000) -> str:
        """Retrieve relevant context for the query."""
        return self.client.context(query, max_chars=max_chars)

    async def process_input(self, user_input: str) -> str:
        """Process user input and generate a demo response."""
        context = await self.get_relevant_context(user_input)

        if context:
            response = (
                f"Based on what I remember:\n{context}\n\n"
                f"Regarding '{user_input}': I understand your question."
            )
        else:
            response = f"I do not have much context about '{user_input}' yet. Tell me more."

        await self.remember_interaction(user_input, response)
        return response

    async def run(self) -> None:
        """Run the agent loop."""
        print("Local Agent with MemoryKernel")
        print("=" * 50)
        print("Type 'exit' to quit, 'memory' to see stats")
        print()

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == "exit":
                    print("Goodbye!")
                    break

                if user_input.lower() == "memory":
                    status = self.client.status()
                    print("\nMemory Stats:")
                    print(f"  Total Memories: {status.total_memories}")
                    print(f"  Total Facts: {status.total_facts}")
                    print(f"  Generation: {status.generation}")
                    print()
                    continue

                response = await self.process_input(user_input)
                print(f"\nAgent: {response}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as exc:
                print(f"\nError: {exc}\n")


async def main() -> None:
    """Run the demo."""
    agent = LocalAgent()

    print("Initializing agent memory...")
    agent.client.remember("I am a helpful AI assistant")
    agent.client.remember("I can remember conversations across sessions")
    agent.client.remember("I use MemoryKernel for persistent memory")
    print("Memory initialized\n")

    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
