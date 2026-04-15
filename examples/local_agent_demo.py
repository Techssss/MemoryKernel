"""
Local Agent with MemoryKernel Integration
==========================================

This example shows how to build a local AI agent that uses MemoryKernel
for persistent memory across sessions.

The agent:
1. Remembers conversations
2. Recalls relevant context
3. Learns from interactions
4. Maintains memory across restarts
"""

import asyncio
from memk.sdk import MemoryKernelClient


class LocalAgent:
    """Simple local agent with persistent memory."""
    
    def __init__(self):
        self.client = MemoryKernelClient()
        self.conversation_history = []
    
    async def remember_interaction(self, user_input: str, agent_response: str):
        """Store interaction in memory."""
        # Remember user input
        await self.client.remember(
            f"User said: {user_input}",
            importance=0.7
        )
        
        # Remember agent response
        await self.client.remember(
            f"Agent responded: {agent_response}",
            importance=0.5
        )
    
    async def get_relevant_context(self, query: str, max_chars: int = 2000) -> str:
        """Retrieve relevant context for the query."""
        result = await self.client.context(query, max_chars=max_chars)
        return result.get("context", "")
    
    async def process_input(self, user_input: str) -> str:
        """Process user input and generate response."""
        # Get relevant context from memory
        context = await self.get_relevant_context(user_input)
        
        # In a real agent, you would call an LLM here
        # For demo purposes, we'll just echo with context
        if context:
            response = f"Based on what I remember:\n{context}\n\nRegarding '{user_input}': I understand your question."
        else:
            response = f"I don't have much context about '{user_input}' yet. Tell me more!"
        
        # Remember this interaction
        await self.remember_interaction(user_input, response)
        
        return response
    
    async def run(self):
        """Run the agent loop."""
        print("🤖 Local Agent with MemoryKernel")
        print("=" * 50)
        print("Type 'exit' to quit, 'memory' to see stats")
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'exit':
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'memory':
                    # Show memory stats
                    status = await self.client.status()
                    stats = status.get("stats", {})
                    print(f"\n📊 Memory Stats:")
                    print(f"  Total Memories: {stats.get('total_memories', 0)}")
                    print(f"  Total Facts: {stats.get('total_active_facts', 0)}")
                    print()
                    continue
                
                # Process input
                response = await self.process_input(user_input)
                print(f"\n🤖 Agent: {response}\n")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


async def main():
    """Run the demo."""
    agent = LocalAgent()
    
    # Add some initial knowledge
    print("📚 Initializing agent memory...")
    await agent.client.remember("I am a helpful AI assistant")
    await agent.client.remember("I can remember conversations across sessions")
    await agent.client.remember("I use MemoryKernel for persistent memory")
    print("✅ Memory initialized\n")
    
    # Run agent loop
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
