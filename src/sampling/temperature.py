"""Temperature and entropy calculations for LLM sampling."""

import numpy as np
from typing import List, Tuple


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Apply softmax with temperature scaling.
    
    Args:
        logits: Raw logit scores
        temperature: Temperature parameter (higher = more random)
        
    Returns:
        Probability distribution
    """
    scaled_logits = logits / temperature
    # Subtract max for numerical stability
    scaled_logits = scaled_logits - np.max(scaled_logits)
    exp_logits = np.exp(scaled_logits)
    return exp_logits / np.sum(exp_logits)


def calculate_entropy(probabilities: np.ndarray) -> float:
    """
    Calculate Shannon entropy of a probability distribution.
    
    Args:
        probabilities: Probability distribution
        
    Returns:
        Entropy in bits
    """
    # Clip to avoid log(0)
    probs = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.sum(probs * np.log2(probs)))


def temperature_entropy_analysis(
    vocab_size: int = 20, 
    temperatures: List[float] = None
) -> Tuple[List[float], List[float]]:
    """
    Analyze how temperature affects output entropy.
    
    Args:
        vocab_size: Size of vocabulary
        temperatures: List of temperature values to test
        
    Returns:
        Tuple of (temperatures, entropies)
    """
    if temperatures is None:
        temperatures = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    
    rng = np.random.default_rng(42)
    logits = rng.normal(size=vocab_size)
    
    entropies = []
    for temp in temperatures:
        probs = softmax(logits, temperature=temp)
        entropy = calculate_entropy(probs)
        entropies.append(entropy)
    
    return temperatures, entropies
