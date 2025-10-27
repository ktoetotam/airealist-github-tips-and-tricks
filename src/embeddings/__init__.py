"""Embeddings package for vector representations and similarity."""

from .embeddings import (
    create_word_embedding,
    normalize_vector,
    cosine_similarity,
    create_similarity_matrix,
)

__all__ = [
    "create_word_embedding",
    "normalize_vector",
    "cosine_similarity",
    "create_similarity_matrix",
]
