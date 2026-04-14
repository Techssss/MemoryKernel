"""
memk.core — Core interfaces, data models, and utilities.
"""

from .embedder import (
    BaseEmbedder,
    SentenceTransformerEmbedder,
    TFIDFEmbedder,
    get_default_embedder,
    cosine_similarity,
    encode_embedding,
    decode_embedding,
)

__all__ = [
    "BaseEmbedder",
    "SentenceTransformerEmbedder",
    "TFIDFEmbedder",
    "get_default_embedder",
    "cosine_similarity",
    "encode_embedding",
    "decode_embedding",
]
