"""Tests for tokenizer module."""

import pytest
from src.tokenization import (
    tokenize,
    get_token_lengths,
    get_token_frequencies,
    get_vocabulary_size
)


def test_tokenize_simple():
    """Test basic tokenization."""
    text = "Hello, World!"
    tokens = tokenize(text)
    assert tokens == ["hello", "world"]


def test_tokenize_empty():
    """Test tokenization of empty string."""
    tokens = tokenize("")
    assert tokens == []


def test_tokenize_numbers():
    """Test tokenization with numbers."""
    text = "Python 3.11 is great!"
    tokens = tokenize(text)
    assert "python" in tokens
    assert "3" in tokens
    assert "11" in tokens


def test_get_token_lengths():
    """Test token length calculation."""
    tokens = ["hello", "world", "ai"]
    lengths = get_token_lengths(tokens)
    assert lengths == [5, 5, 2]


def test_get_token_frequencies():
    """Test frequency counting."""
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    frequencies = get_token_frequencies(tokens, top_k=3)
    assert frequencies[0] == ("the", 2)
    assert len(frequencies) == 3


def test_get_vocabulary_size():
    """Test vocabulary size calculation."""
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    vocab_size = get_vocabulary_size(tokens)
    assert vocab_size == 5  # "the", "cat", "sat", "on", "mat"
