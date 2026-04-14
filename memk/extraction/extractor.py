from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel
import re

class StructuredFact(BaseModel):
    """A minimal triplet representing structured knowledge."""
    subject: str
    relation: str
    object: str

class BaseExtractor(ABC):
    """
    Abstract base class for fact extraction.
    This contract ensures that future LLM-based extractors can drop-in replace
    rule-based ones without altering the orchestrator logic.
    """
    @abstractmethod
    def extract_facts(self, text: str) -> List[StructuredFact]:
        """Convert raw text to structured triplets."""
        pass

class RuleBasedExtractor(BaseExtractor):
    """
    Naive MVP extractor using regular expressions.
    Designed to catch exact patterns. Ignores conversational noise.
    """
    def __init__(self):
        # Basic sentence structure matching: Subject + Verb/Relation + Object
        # Focus on standardizing English/Vietnamese developer-specific terms
        self.patterns = [
            re.compile(r'(?i)^(user|project|tao|tôi|hệ thống|system)\s+(thích|ghét|dùng|sử dụng|cần|muốn|yêu cầu|likes|uses|hates|needs|wants|requires)\s+(.+)$')
        ]

    def _normalize_relation(self, verb: str) -> str:
        """Standardize synonyms into a canonical dictionary of relations."""
        mapping = {
            "thích": "likes",
            "ghét": "dislikes",
            "dùng": "uses",
            "sử dụng": "uses",
            "cần": "needs",
            "muốn": "wants",
            "yêu cầu": "requires",
            "hates": "dislikes"
        }
        return mapping.get(verb.lower(), verb.lower())
    
    def _normalize_subject(self, subj: str) -> str:
        mapping = {
            "tôi": "user",
            "tao": "user",
            "hệ thống": "system",
            "project": "project",
            "user": "user"
        }
        return mapping.get(subj.lower(), subj.lower())

    def extract_facts(self, text: str) -> List[StructuredFact]:
        facts = []
        # Split input into semi-independent text blocks (roughly by sentences/clauses)
        clauses = re.split(r'[;,\.\n]+', text)
        
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
                
            for pattern in self.patterns:
                match = pattern.search(clause)
                if match:
                    subj, rel, obj = match.groups()
                    facts.append(StructuredFact(
                        subject=self._normalize_subject(subj),
                        relation=self._normalize_relation(rel),
                        object=obj.strip()
                    ))
                    break  # Once a pattern fires, stop looking for rules in this specific clause
                    
        return facts
