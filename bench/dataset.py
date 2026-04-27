import random
import hashlib
from typing import List, Dict, Any

class SyntheticDataset:
    """
    Generates deterministic synthetic data for MemoryKernel benchmarks.
    Includes support for clusters, multi-hop facts, and noise.
    """
    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)
        self.topics = [
            "Quantum Computing", "Ancient Rome", "Baking Sourdough", 
            "Autonomous Vehicles", "Space Exploration", "Gardening",
            "Functional Programming", "Renaissance Art", "Climate Change"
        ]
        self.names = ["Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona", "George", "Hannah"]
        self.companies = ["TechCorp", "GlobalLogistics", "BioNano", "AeroSystems", "FutureAI"]
        self.cities = ["Tokyo", "Berlin", "San Francisco", "London", "Paris"]

    def generate_memories(self, count: int) -> List[Dict[str, Any]]:
        memories = []
        for i in range(count):
            topic = self.random.choice(self.topics)
            # Mix of lengths
            length = self.random.choice(["short", "medium", "long"])
            if length == "short":
                content = f"Note about {topic}: {self._rand_str(10)}"
            elif length == "medium":
                content = f"Discussion on {topic}. {self._rand_str(50)}. Importance is high."
            else:
                content = f"Deep dive into {topic}. " + " ".join([self._rand_str(15) for _ in range(10)])
            
            memories.append({
                "id": f"mem_{i}_{hashlib.md5(content.encode()).hexdigest()[:8]}",
                "content": content,
                "topic": topic
            })
        return memories

    def generate_facts(self, count: int) -> List[Dict[str, Any]]:
        facts = []
        for i in range(count):
            # Mix of isolated and connected facts
            type_choice = self.random.random()
            if type_choice < 0.3: # Multi-hop: Person -> Company
                name = self.random.choice(self.names)
                comp = self.random.choice(self.companies)
                subj = name
                obj = f"works at {comp}"
                meta = {"type": "employment"}
            elif type_choice < 0.6: # Multi-hop: Company -> Location
                comp = self.random.choice(self.companies)
                city = self.random.choice(self.cities)
                subj = comp
                obj = f"is located in {city}"
                meta = {"type": "location"}
            else:
                subj = self._rand_str(5)
                obj = f"attribute {self._rand_str(10)}"
                meta = {"type": "general"}

            facts.append({
                "id": f"fact_{i}",
                "subject": subj,
                "object": obj,
                "metadata": meta
            })
        return facts

    def generate_graph_items(self, memory_count: int) -> List[Dict[str, Any]]:
        """Produces entities and mentions for existing memories."""
        # Simple entity generation based on topics and names
        entities = []
        for name in self.names:
            entities.append({"id": name, "type": "person"})
        for comp in self.companies:
            entities.append({"id": comp, "type": "org"})
        for city in self.cities:
            entities.append({"id": city, "type": "place"})
        return entities

    def _rand_str(self, length: int) -> str:
        chars = "abcdefghijklmnopqrstuvwxyz "
        return "".join(self.random.choice(chars) for _ in range(length))
