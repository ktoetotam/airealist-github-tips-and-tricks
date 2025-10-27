"""Simple tokenizer for text processing."""

import re
from typing import List
from collections import Counter


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words using regex.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of lowercase tokens
        
    Examples:
        >>> tokenize("Hello, World!")
        ['hello', 'world']
    """
    return [token.lower() for token in re.findall(r"\w+", text)]


def get_token_lengths(tokens: List[str]) -> List[int]:
    """
    Calculate the length of each token.
    
    Args:
        tokens: List of tokens
        
    Returns:
        List of token lengths
    """
    return [len(token) for token in tokens]


def get_token_frequencies(tokens: List[str], top_k: int = 10) -> List[tuple]:
    """
    Get the most common tokens with their frequencies.
    
    Args:
        tokens: List of tokens
        top_k: Number of top tokens to return
        
    Returns:
        List of (token, count) tuples sorted by frequency
    """
    counter = Counter(tokens)
    return counter.most_common(top_k)


def get_vocabulary_size(tokens: List[str]) -> int:
    """
    Calculate unique vocabulary size.
    
    Args:
        tokens: List of tokens
        
    Returns:
        Number of unique tokens
    """
    return len(set(tokens))
