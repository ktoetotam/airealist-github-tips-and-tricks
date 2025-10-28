# LLM Visualization Toolkit - AI Agent Instructions

## Project Overview
Educational Python package that visualizes Large Language Model concepts through matplotlib. Creates 4-panel figures showing tokenization, embeddings, temperature effects, and similarity analysis.

## Architecture & Key Files

### Entry Point & CLI
- **Main**: `src/llm_plots.py` - CLI with `--text`, `--file`, `--out` args
- **Script Command**: `llm-plots` (defined in pyproject.toml entry points)
- **Demo Text**: Built-in educational text about LLMs in `get_demo_text()`

### Core Modules (src-layout package)
```
src/
├── tokenization/    # Regex-based text splitting, frequency analysis
├── embeddings/      # Hash-seeded deterministic vectors, cosine similarity  
├── sampling/        # Temperature scaling, entropy calculations
├── visualization/   # matplotlib 4-panel figure generation
└── llm_plots.py    # CLI entry point
```

### Module Import Pattern
- Use relative imports: `from ..tokenization import tokenize`
- Each module has `__init__.py` exposing key functions
- Main visualization calls across all modules in `create_llm_visualization()`

## Development Workflow

### Testing (pytest)
- **Run**: `pytest tests/` or `pytest tests/test_specific.py`
- **Coverage**: `pytest --cov=src --cov-report=html`
- **Pattern**: Test deterministic outputs (embeddings, tokenization)
- **Key Assertions**: `np.testing.assert_array_equal()` for numpy arrays

### Code Quality (configured in pyproject.toml)
- **Format**: `black .` (100 char line length)
- **Lint**: `flake8 src/ tests/`
- **Types**: `mypy src/`
- **Install dev tools**: `pip install -e ".[dev]"`

### Package Installation
- **Development**: `pip install -e .` (editable install)
- **Production**: `pip install -r requirements.txt`

## Critical Patterns

### Visualization Flow
1. Text → `tokenize()` → `get_token_frequencies()`
2. Top words → `create_similarity_matrix()` 
3. Temperature range → `temperature_entropy_analysis()`
4. All data → `create_llm_visualization()` → 4-panel matplotlib figure

### Deterministic Design
- **Embeddings**: Hash-seeded for reproducible vectors
- **Testing**: Assert exact equality, not approximations
- **Random elements**: Use fixed seeds (see `create_word_embedding()`)

### Error Handling
- Check file existence before reading (`Path.exists()`)
- Validate non-empty token lists in visualization
- Create output directories with `os.makedirs(exist_ok=True)`

## Common Operations

### Adding New Visualizations
1. Create function in `src/visualization/plots.py`
2. Accept `ax=None` parameter for subplot integration
3. Import and call in `create_llm_visualization()`
4. Add corresponding test in `tests/test_*.py`

### Adding New Analysis Functions
1. Place in appropriate module (`tokenization/`, `embeddings/`, `sampling/`)
2. Export in module's `__init__.py`
3. Write comprehensive tests with edge cases
4. Document parameters and return types

### Testing New Features
- Test both valid and edge cases (empty strings, single tokens)
- Use `np.testing.assert_*` for numerical comparisons
- Ensure deterministic outputs for reproducibility
- Test matplotlib figure creation without displaying


If you are not sure, do not guess, ask for clarification!