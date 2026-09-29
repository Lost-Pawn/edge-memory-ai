"""On-device embedding providers."""

from edge_memory.embeddings.base import EmbeddingProvider, SparseVector
from edge_memory.embeddings.fastembed_provider import FastEmbedProvider

__all__ = ["EmbeddingProvider", "FastEmbedProvider", "SparseVector"]
