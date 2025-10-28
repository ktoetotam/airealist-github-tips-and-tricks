---
description: Testing guidelines for LLM Visualization Toolkit using pytest with focus on deterministic assertions and edge case coverage
applyTo: tests/**
---

# Tests Instructions

## Context
These instructions apply when working with test files in the `tests/` directory. This project uses pytest for testing all core functionality including tokenization, embeddings, sampling, and visualization modules.

## Code Patterns

### 1. Test Function Naming
- **Pattern**: `test_<module>_<scenario>()`
- **Example**: `test_tokenize_empty()`, `test_embedding_deterministic()`
```python
def test_tokenize_simple():
    """Test basic tokenization."""
    text = "Hello, World!"
    tokens = tokenize(text)
    assert tokens == ["hello", "world"]
```

### 2. Docstring Convention
- Every test function must have a docstring
- Format: Brief description of what is being tested
```python
def test_cosine_similarity_identical():
    """Test cosine similarity of identical vectors."""
    # test implementation
```

### 3. Import Pattern
- Import pytest at the top
- Import specific functions from source modules using absolute imports
```python
import pytest
import numpy as np
from src.tokenization import tokenize, get_token_frequencies
from src.embeddings import create_word_embedding, cosine_similarity
```

### 4. Arrange-Act-Assert Structure
```python
def test_get_token_lengths():
    """Test token length calculation."""
    # Arrange
    tokens = ["hello", "world", "ai"]
    
    # Act
    lengths = get_token_lengths(tokens)
    
    # Assert
    assert lengths == [5, 5, 2]
```

## Best Practices

### 1. Deterministic Testing
- **Always test that deterministic functions produce identical results**
- Use `np.testing.assert_array_equal()` for exact numpy array comparisons
```python
def test_embedding_deterministic():
    """Test that embeddings are deterministic."""
    emb1 = create_word_embedding("test", dimensions=10)
    emb2 = create_word_embedding("test", dimensions=10)
    np.testing.assert_array_equal(emb1, emb2)
```

### 2. Edge Case Coverage
- **Empty inputs**: Test with empty strings, empty lists
- **Single elements**: Test with one token, one word
- **Boundary values**: Test with minimum/maximum valid inputs
```python
def test_tokenize_empty():
    """Test tokenization of empty string."""
    tokens = tokenize("")
    assert tokens == []

def test_calculate_entropy_deterministic():
    """Test entropy of deterministic distribution."""
    probs = np.array([1.0, 0.0, 0.0, 0.0])
    entropy = calculate_entropy(probs)
    assert np.isclose(entropy, 0.0)
```

### 3. Numerical Comparisons
- Use `np.isclose()` for floating-point comparisons with tolerance
- Use `np.allclose()` for array-wise comparisons
- Use exact equality only for deterministic integer/string outputs
```python
def test_normalize_vector():
    """Test vector normalization."""
    vector = np.array([3.0, 4.0])
    normalized = normalize_vector(vector)
    assert np.isclose(np.linalg.norm(normalized), 1.0)
    assert np.isclose(normalized[0], 0.6)
```

### 4. Test Mathematical Properties
- Symmetry: `assert np.allclose(matrix, matrix.T)`
- Sum to one: `assert np.isclose(np.sum(probs), 1.0)`
- Self-similarity: `assert np.allclose(np.diag(matrix), 1.0)`
```python
def test_create_similarity_matrix():
    """Test similarity matrix creation."""
    words = ["hello", "world", "test"]
    matrix = create_similarity_matrix(words, dimensions=8)
    # Matrix should be symmetric
    assert np.allclose(matrix, matrix.T)
    # Diagonal should be 1.0 (self-similarity)
    assert np.allclose(np.diag(matrix), 1.0)
```

## Common Tasks

### Adding a New Test
1. Create function following `test_<function>_<scenario>` naming
2. Add docstring describing what is tested
3. Use arrange-act-assert structure
4. Test both happy path and edge cases
5. Run pytest to verify: `pytest tests/test_module.py`

### Testing a New Module Function
1. Import the function from the appropriate src module
2. Test return types and shapes (for numpy arrays)
3. Test deterministic behavior if applicable
4. Test edge cases (empty, single element, large inputs)
5. Test mathematical properties if relevant

### Testing Visualization Functions
1. Check that matplotlib figures/axes are created
2. Verify correct number of plot elements
3. Test with minimal valid data
4. Don't assert on exact pixel values, test structure instead
```python
def test_plot_creation():
    """Test that plot is created successfully."""
    fig, ax = plt.subplots()
    plot_token_frequencies(tokens, ax=ax)
    assert len(ax.patches) > 0  # Check bars were created
```

## Error Handling

### Expected Exceptions
- Use `pytest.raises()` for testing error conditions
```python
def test_invalid_temperature():
    """Test that negative temperature raises error."""
    with pytest.raises(ValueError):
        softmax(logits, temperature=-1.0)
```

### Invalid Inputs
- Test functions handle empty inputs gracefully
- Test boundary conditions (zero, negative values)
- Verify appropriate error messages

## Testing Guidelines

### Test Organization
- One test file per source module (`test_tokenizer.py` for `tokenizer.py`)
- Group related tests together
- Use descriptive test names that explain the scenario

### Test Independence
- Each test should run independently
- Don't rely on test execution order
- Clean up any state changes (though this project has minimal state)

### Coverage Goals
- Aim for 100% code coverage of source modules
- Every public function should have at least 2 tests (happy path + edge case)
- Run coverage report: `pytest --cov=src --cov-report=html`

## Key Dependencies

### pytest (>=7.4.0)
- Test runner and assertion framework
- Run with: `pytest tests/`
- Verbose mode: `pytest -v tests/`

### pytest-cov (>=4.1.0)
- Coverage reporting
- Usage: `pytest --cov=src --cov-report=term-missing`
- HTML report: `pytest --cov=src --cov-report=html`

### numpy.testing
- Specialized assertions for numerical arrays
- `assert_array_equal()`: Exact equality
- `assert_allclose()`: Approximate equality with tolerance
```python
import numpy as np
np.testing.assert_array_equal(actual, expected)
np.testing.assert_allclose(actual, expected, rtol=1e-7)
```

## Examples

### Complete Test Example
```python
"""Tests for tokenizer module."""

import pytest
from src.tokenization import tokenize, get_token_frequencies

def test_tokenize_simple():
    """Test basic tokenization."""
    # Arrange
    text = "Hello, World!"
    
    # Act
    tokens = tokenize(text)
    
    # Assert
    assert tokens == ["hello", "world"]
    assert len(tokens) == 2

def test_tokenize_empty():
    """Test tokenization of empty string."""
    tokens = tokenize("")
    assert tokens == []
    
def test_get_token_frequencies():
    """Test frequency counting."""
    # Arrange
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    
    # Act
    frequencies = get_token_frequencies(tokens, top_k=3)
    
    # Assert
    assert frequencies[0] == ("the", 2)
    assert len(frequencies) == 3
    assert all(isinstance(item, tuple) for item in frequencies)
```

### Testing with NumPy Arrays
```python
import numpy as np
from src.embeddings import create_word_embedding, cosine_similarity

def test_embedding_deterministic():
    """Test that embeddings are deterministic."""
    emb1 = create_word_embedding("test", dimensions=10)
    emb2 = create_word_embedding("test", dimensions=10)
    np.testing.assert_array_equal(emb1, emb2)

def test_cosine_similarity_orthogonal():
    """Test cosine similarity of orthogonal vectors."""
    vec_a = np.array([1.0, 0.0])
    vec_b = np.array([0.0, 1.0])
    similarity = cosine_similarity(vec_a, vec_b)
    assert np.isclose(similarity, 0.0, atol=1e-7)
```

### Testing Distribution Properties
```python
import numpy as np
from src.sampling import softmax, calculate_entropy

def test_softmax_sum_to_one():
    """Test that softmax outputs sum to 1."""
    logits = np.array([1.0, 2.0, 3.0])
    probs = softmax(logits)
    assert np.isclose(np.sum(probs), 1.0)

def test_calculate_entropy_uniform():
    """Test entropy of uniform distribution."""
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    entropy = calculate_entropy(probs)
    assert np.isclose(entropy, 2.0)  # log2(4) = 2
```
