import os
import logging
from typing import List, Dict, Any, Optional
from memk.storage.db import MemoryDB

logger = logging.getLogger(__name__)

class KnowledgeSynthesizer:
    """
    Transforms structured database facts into human-readable Markdown pages.
    Organizes knowledge by subject and hierarchies.
    """

    def __init__(self, db: MemoryDB, output_dir: str = "knowledge"):
        self.db = db
        self.output_dir = output_dir

    def synthesize_all(self) -> List[str]:
        """
        Generate Markdown pages for every unique subject in the system,
        plus an index.md linking them all together.
        Returns a list of generated file paths.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        subjects = self.db.get_all_subjects()
        generated_files = []

        # 1. Generate topic pages
        for subject in subjects:
            filepath = self.synthesize_topic(subject)
            if filepath:
                generated_files.append(filepath)

        # 2. Generate index.md
        index_path = os.path.join(self.output_dir, "index.md")
        self._write_index(subjects, index_path)
        generated_files.append(index_path)

        return generated_files

    def synthesize_topic(self, subject: str) -> Optional[str]:
        """
        Generate a single Markdown page for a specific subject.
        Groups facts by predicate to build a hierarchy.
        """
        facts = self.db.search_facts(subject=subject)
        if not facts:
            logger.warning(f"No facts found for subject: {subject}")
            return None

        # Group by predicate
        grouped: Dict[str, List[str]] = {}
        for f in facts:
            pred = f["predicate"]
            obj = f["object"]
            if pred not in grouped:
                grouped[pred] = []
            grouped[pred].append(obj)

        # Build Markdown content
        lines = [f"# {subject.title()}\n"]
        
        # Sort predicates deterministically
        for pred in sorted(grouped.keys()):
            lines.append(f"## {pred.title()}")
            # Sort objects deterministically
            for obj in sorted(grouped[pred]):
                lines.append(f"* {obj}")
            lines.append("")  # Spacer

        # Write to file
        safe_name = "".join(c if c.isalnum() else "_" for c in subject.lower())
        filename = f"{safe_name}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).strip() + "\n")

        return filepath

    def _write_index(self, subjects: List[str], filepath: str):
        """Build the master index page."""
        lines = ["# Knowledge Index\n"]
        lines.append("Welcome to the synthesized knowledge base of MemoryKernel.\n")

        for subject in sorted(subjects):
            safe_name = "".join(c if c.isalnum() else "_" for c in subject.lower())
            lines.append(f"* [{subject.title()}]({safe_name}.md)")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).strip() + "\n")
