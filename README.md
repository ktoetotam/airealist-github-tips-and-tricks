# LLM Visualization Toolkit

A simple Python project demonstrating LLM concepts through matplotlib visualizations. Perfect for learning GitHub Copilot!

## Features

- 📊 Token analysis and visualization
- 🔥 Temperature vs entropy plots
- 🎯 Cosine similarity heatmaps
- 📈 Word frequency distributions

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run with default demo text
python src/llm_plots.py

# Analyze custom text
python src/llm_plots.py --text "Your custom text here"

# Analyze a file
python src/llm_plots.py --file path/to/text.txt --out output/plot.png
```

## Run Tests

```bash
pytest tests/
```

## Project Structure

```
├── src/              # Source code
├── tests/            # Unit tests
├── data/             # Sample data files
├── output/           # Generated plots
└── notebooks/        # Jupyter notebooks for exploration
```
