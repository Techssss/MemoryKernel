"""
memk.ingestion
==============
Knowledge ingestion from external sources (Git, files, etc.)
"""

from .git_ingestor import GitIngestor, CommitMetadata, IngestionRule

__all__ = ["GitIngestor", "CommitMetadata", "IngestionRule"]
