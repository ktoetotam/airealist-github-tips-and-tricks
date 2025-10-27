"""Main entry point for LLM visualization toolkit."""

import argparse
from pathlib import Path

from .visualization import create_llm_visualization


def get_demo_text() -> str:
    """Return demo text about LLMs."""
    return (
        "Large language models generate text token by token using neural networks. "
        "Temperature controls the randomness of sampling from the probability distribution. "
        "Higher temperature values increase entropy and produce more creative outputs. "
        "Word embeddings map tokens to high-dimensional vectors for semantic similarity. "
        "Attention mechanisms allow models to focus on relevant context when generating text. "
        "Tokenization splits text into subword units that the model can process efficiently. "
        "The softmax function converts raw logits into probability distributions over vocabulary."
    )


def main():
    """Main function to run the LLM visualization toolkit."""
    parser = argparse.ArgumentParser(
        description="Generate LLM concept visualizations using matplotlib"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Inline text to analyze"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a text file to analyze"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output/llm_plots.png",
        help="Output image path"
    )
    
    args = parser.parse_args()
    
    # Get text from file or argument or use demo
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File '{args.file}' not found")
            return
        text = file_path.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        text = get_demo_text()
        print("Using demo text (use --text or --file to provide custom input)")
    
    # Create visualization
    try:
        output_path = create_llm_visualization(text, args.out)
        print(f"✓ Visualization saved to: {output_path}")
    except Exception as e:
        print(f"Error creating visualization: {e}")
        raise


if __name__ == "__main__":
    main()
