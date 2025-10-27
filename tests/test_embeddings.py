"""Tests for embeddings module."""

import pytest
import numpy as np
from src.embeddings import (
    create_word_embedding,
    normalize_vector,
    cosine_similarity,
    create_similarity_matrix
)


def test_create_word_embedding():
    """Test word embedding creation."""
    embedding = create_word_embedding("hello", dimensions=16)
    assert embedding.shape == (16,)
    assert np.isclose(np.linalg.norm(embedding), 1.0)


def test_embedding_deterministic():
    """Test that embeddings are deterministic."""
    emb1 = create_word_embedding("test", dimensions=10)
    emb2 = create_word_embedding("test", dimensions=10)
    np.testing.assert_array_equal(emb1, emb2)


def test_normalize_vector():
    """Test vector normalization."""
    vector = np.array([3.0, 4.0])
    normalized = normalize_vector(vector)
    assert np.isclose(np.linalg.norm(normalized), 1.0)
    assert np.isclose(normalized[0], 0.6)
    assert np.isclose(normalized[1], 0.8)


def test_cosine_similarity_identical():
    """Test cosine similarity of identical vectors."""
    vec = np.array([1.0, 0.0])
    similarity = cosine_similarity(vec, vec)
    assert np.isclose(similarity, 1.0)


def test_cosine_similarity_orthogonal():
    """Test cosine similarity of orthogonal vectors."""
    vec_a = np.array([1.0, 0.0])
    vec_b = np.array([0.0, 1.0])
    similarity = cosine_similarity(vec_a, vec_b)
    assert np.isclose(similarity, 0.0)


def test_create_similarity_matrix():
    """Test similarity matrix creation."""
    words = ["hello", "world", "test"]
    matrix = create_similarity_matrix(words, dimensions=8)
    assert matrix.shape == (3, 3)
    # Diagonal should be 1.0 (self-similarity)
    assert np.allclose(np.diag(matrix), 1.0)
    # Matrix should be symmetric
    assert np.allclose(matrix, matrix.T)
