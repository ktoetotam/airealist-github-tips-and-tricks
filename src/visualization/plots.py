"""Visualization functions for LLM concepts."""

import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def plot_token_length_distribution(lengths: List[int], ax=None):
    """
    Plot histogram of token lengths.
    
    Args:
        lengths: List of token lengths
        ax: Matplotlib axis (creates new if None)
    """
    if ax is None:
        _, ax = plt.subplots()
    
    ax.hist(lengths, bins=range(1, max(lengths) + 2), color="#4C78A8", edgecolor="white")
    ax.set_title("Token Length Distribution")
    ax.set_xlabel("Token length (chars)")
    ax.set_ylabel("Count")


def plot_token_frequencies(frequencies: List[Tuple[str, int]], ax=None):
    """
    Plot bar chart of token frequencies.
    
    Args:
        frequencies: List of (token, count) tuples
        ax: Matplotlib axis (creates new if None)
    """
    if ax is None:
        _, ax = plt.subplots()
    
    tokens = [token for token, _ in frequencies]
    counts = [count for _, count in frequencies]
    
    ax.bar(tokens, counts, color="#F58518")
    ax.set_title(f"Top-{len(tokens)} Tokens")
    ax.set_ylabel("Frequency")
    ax.set_xticklabels(tokens, rotation=30, ha="right")


def plot_similarity_heatmap(similarity_matrix: np.ndarray, labels: List[str], ax=None):
    """
    Plot cosine similarity heatmap.
    
    Args:
        similarity_matrix: NxN similarity matrix
        labels: Token labels for axes
        ax: Matplotlib axis (creates new if None)
    """
    if ax is None:
        _, ax = plt.subplots()
    
    im = ax.imshow(similarity_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("Token Similarity Heatmap")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_temperature_entropy(temperatures: List[float], entropies: List[float], ax=None):
    """
    Plot temperature vs entropy curve.
    
    Args:
        temperatures: List of temperature values
        entropies: List of entropy values
        ax: Matplotlib axis (creates new if None)
    """
    if ax is None:
        _, ax = plt.subplots()
    
    ax.plot(temperatures, entropies, marker="o", color="#54A24B", linewidth=2)
    ax.set_title("Temperature vs Output Entropy")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Entropy (bits)")
    ax.grid(True, alpha=0.3)


def create_llm_visualization(text: str, output_path: str = "output/llm_plots.png"):
    """
    Create comprehensive LLM visualization from text.
    
    Args:
        text: Input text to analyze
        output_path: Path to save the output image
        
    Returns:
        Path to the saved image
    """
    from ..tokenization import tokenize, get_token_lengths, get_token_frequencies
    from ..embeddings import create_similarity_matrix
    from ..sampling import temperature_entropy_analysis
    
    # Tokenize and analyze
    tokens = tokenize(text)
    if not tokens:
        raise ValueError("No tokens found in input text")
    
    lengths = get_token_lengths(tokens)
    frequencies = get_token_frequencies(tokens, top_k=10)
    top_words = [word for word, _ in frequencies]
    similarity_matrix = create_similarity_matrix(top_words, dimensions=16)
    temps, entropies = temperature_entropy_analysis(vocab_size=30)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("LLM Concepts Visualization", fontsize=14, fontweight="bold")
    
    # Plot each visualization
    ax1 = plt.subplot(2, 2, 1)
    plot_token_length_distribution(lengths, ax=ax1)
    
    ax2 = plt.subplot(2, 2, 2)
    plot_token_frequencies(frequencies, ax=ax2)
    
    ax3 = plt.subplot(2, 2, 3)
    plot_similarity_heatmap(similarity_matrix, top_words, ax=ax3)
    
    ax4 = plt.subplot(2, 2, 4)
    plot_temperature_entropy(temps, entropies, ax=ax4)
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return output_path
