#!/usr/bin/env python
"""
Agent with Memory Example
=========================
Demonstrates integrating MemoryKernel into an AI agent.
"""

from memk.sdk import MemoryKernel
from typing import Optional

class MemoryAgent:
    """Simple agent with persistent memory."""
    
    def __init__(self):
        self.memory = MemoryKernel()
        print("Agent initialized with memory")
    
    def process(self, user_input: str) -> str:
        """
        Process user input with memory context.
        
        1. Retrieve relevant context from memory
        2. Generate response (simulated)
        3. Remember the interaction
        """
        print(f"\n[User]: {user_input}")
        
        # Get relevant context
        context = self.memory.context(user_input, max_chars=1000)
        
        if context:
            print(f"[Memory Context]: {context[:100]}...")
        
        # Simulate LLM response (in real app, use actual LLM)
        response = self._generate_response(user_input, context)
        
        # Remember this interaction
        self.memory.remember(
            f"User asked: {user_input}",
            importance=0.6
        )
        
        print(f"[Agent]: {response}")
        return response
    
    def _generate_response(self, user_input: str, context: str) -> str:
        """Simulate LLM response generation."""
        # In real app, call LLM with context
        if "prefer" in user_input.lower():
            return "Based on your preferences, I recommend TypeScript."
        elif "database" in user_input.lower():
            return "The project uses PostgreSQL as the database."
        else:
            return "I'll help you with that based on what I know about the project."
    
    def learn(self, fact: str, importance: float = 0.7):
        """Explicitly teach the agent something."""
        self.memory.remember(fact, importance=importance)
        print(f"[Learned]: {fact}")

def main():
    print("=== Memory Agent Demo ===")
    
    # Create agent
    agent = MemoryAgent()
    
    # Teach agent some facts
    print("\n--- Teaching Agent ---")
    agent.learn("User is building a web application", importance=0.8)
    agent.learn("User prefers functional programming", importance=0.7)
    agent.learn("Project deadline is end of month", importance=0.9)
    
    # Interact with agent
    print("\n--- Conversations ---")
    agent.process("What programming style should I use?")
    agent.process("What database are we using?")
    agent.process("When is the deadline?")
    
    # Check memory status
    print("\n--- Memory Status ---")
    status = agent.memory.status()
    print(f"Total memories: {status.total_memories}")
    print(f"Generation: {status.generation}")
    
    print("\n✓ Demo complete!")

if __name__ == "__main__":
    main()
