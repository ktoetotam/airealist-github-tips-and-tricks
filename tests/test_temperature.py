"""Tests for temperature module."""

import pytest
import numpy as np
from src.sampling import (
    softmax,
    calculate_entropy,
    temperature_entropy_analysis
)


def test_softmax_sum_to_one():
    """Test that softmax outputs sum to 1."""
    logits = np.array([1.0, 2.0, 3.0])
    probs = softmax(logits)
    assert np.isclose(np.sum(probs), 1.0)


def test_softmax_temperature_effect():
    """Test that higher temperature flattens distribution."""
    logits = np.array([1.0, 2.0, 5.0])
    low_temp = softmax(logits, temperature=0.5)
    high_temp = softmax(logits, temperature=2.0)
    # High temp should have more uniform distribution
    assert np.std(high_temp) < np.std(low_temp)


def test_calculate_entropy_uniform():
    """Test entropy of uniform distribution."""
    # Uniform distribution has maximum entropy
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    entropy = calculate_entropy(probs)
    assert np.isclose(entropy, 2.0)  # log2(4) = 2


def test_calculate_entropy_deterministic():
    """Test entropy of deterministic distribution."""
    # Deterministic distribution has zero entropy
    probs = np.array([1.0, 0.0, 0.0, 0.0])
    entropy = calculate_entropy(probs)
    assert np.isclose(entropy, 0.0)


def test_temperature_entropy_analysis():
    """Test temperature-entropy relationship."""
    temps, entropies = temperature_entropy_analysis(vocab_size=20)
    assert len(temps) == len(entropies)
    # Entropy should generally increase with temperature
    assert entropies[-1] > entropies[0]


def test_temperature_entropy_custom_temps():
    """Test with custom temperature values."""
    custom_temps = [0.1, 0.5, 1.0, 2.0]
    temps, entropies = temperature_entropy_analysis(
        vocab_size=10, 
        temperatures=custom_temps
    )
    assert temps == custom_temps
    assert len(entropies) == len(custom_temps)
