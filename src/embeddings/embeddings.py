"""Word embeddings and similarity calculations."""

import numpy as np
from typing import List


def create_word_embedding(word: str, dimensions: int = 16) -> np.ndarray:
    """
    Create a deterministic pseudo-embedding for a word.
    
    Args:
        word: The word to embed
        dimensions: Embedding dimension size
        
    Returns:
        Normalized embedding vector
    """
    seed = abs(hash(word)) % (2**32)
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dimensions)
    return normalize_vector(vector)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Args:
        vector: Input vector
        
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec_a: First vector
        vec_b: Second vector
        
    Returns:
        Cosine similarity score [-1, 1]
    """
    return float(np.dot(vec_a, vec_b))


def create_similarity_matrix(words: List[str], dimensions: int = 16) -> np.ndarray:
    """
    Create a similarity matrix for a list of words.
    
    Args:
        words: List of words
        dimensions: Embedding dimension size
        
    Returns:
        NxN similarity matrix where N = len(words)
    """
    embeddings = np.stack([create_word_embedding(w, dimensions) for w in words], axis=0)
    return embeddings @ embeddings.T
