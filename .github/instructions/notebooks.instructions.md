---
description: Interactive development guidelines for Jupyter notebooks focusing on LLM visualization exploration, reproducible analysis, and educational demonstrations
applyTo: notebooks/**
---

# Notebooks Instructions

## Context
These instructions apply when working with Jupyter notebooks in the `notebooks/` directory. Notebooks are used for interactive exploration, visualization experiments, and educational demonstrations of LLM concepts. They provide a sandbox for testing functions and creating visual explanations.

## Code Patterns

### 1. Notebook Structure
- **Pattern**: Start with markdown introduction, then imports, then sections with analysis
- **Example**:
```markdown
# LLM Visualization Exploration
Interactive notebook for exploring LLM concepts through visualizations.
```
```python
# Import block with path setup
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from src.tokenization import tokenize, get_token_frequencies
from src.embeddings import create_similarity_matrix

%matplotlib inline
```

### 2. Path Setup for Module Imports
- **Pattern**: Add parent directory to sys.path to import from src/
- **Critical**: Always include this at the start of notebooks
```python
import sys
sys.path.append('..')  # Access parent directory for src imports

from src.tokenization import tokenize
from src.embeddings import create_word_embedding
```

### 3. Section Headers with Markdown
- **Pattern**: Use markdown cells to create clear section breaks
- **Format**: Use ## for major sections, ### for subsections
```markdown
## 1. Tokenization Example
Demonstrating how text is split into tokens.

## 2. Temperature vs Entropy
Exploring the relationship between temperature and output randomness.
```

### 4. Inline Plotting with matplotlib
- **Pattern**: Always include `%matplotlib inline` magic command
- **Usage**: Create figures inline for immediate visualization
```python
%matplotlib inline

plt.figure(figsize=(8, 5))
plt.plot(temps, entropies, marker='o', linewidth=2)
plt.xlabel('Temperature')
plt.ylabel('Entropy (bits)')
plt.title('Temperature vs Output Entropy')
plt.grid(True, alpha=0.3)
plt.show()
```

### 5. Interactive Experimentation Cells
- **Pattern**: Provide template cells for users to modify
- **Include**: Comments encouraging experimentation
```python
# Experiment with your own text here!
custom_text = "..."
custom_tokens = tokenize(custom_text)
frequencies = get_token_frequencies(custom_tokens, top_k=10)
```

## Best Practices

### 1. Cell Organization
- **One concept per cell**: Keep cells focused on single tasks
- **Markdown before code**: Explain what the next code cell will do
- **Print outputs**: Use print() or display results to show intermediate steps
```python
text = "Large language models generate text token by token."
tokens = tokenize(text)
print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")
```

### 2. Reproducible Results
- **Set random seeds**: Ensure deterministic behavior for educational purposes
- **Document parameters**: Clearly show what values are being used
- **Fixed examples**: Use consistent example text across sections
```python
# Using fixed vocabulary size for reproducibility
temps, entropies = temperature_entropy_analysis(vocab_size=30)
```

### 3. Visual Clarity
- **Figure sizing**: Always specify figsize for consistent displays
- **Labels and titles**: Include descriptive axis labels and titles
- **Legends and colorbars**: Add explanatory elements
- **Grid lines**: Use subtle grids for easier reading
```python
plt.figure(figsize=(8, 6))
plt.imshow(sim_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Cosine Similarity')
plt.xticks(range(len(words)), words, rotation=45, ha='right')
plt.yticks(range(len(words)), words)
plt.title('Word Similarity Heatmap')
plt.tight_layout()
plt.show()
```

### 4. Educational Narrative
- **Tell a story**: Structure notebook as a learning journey
- **Explain concepts**: Use markdown to explain what each visualization shows
- **Connect ideas**: Link sections to show how concepts relate
```markdown
## 3. Word Similarity Matrix
Word embeddings allow us to measure semantic similarity. 
Let's visualize how related different LLM-related terms are to each other.
```

### 5. Error Prevention
- **Check imports**: Verify all modules are accessible
- **Validate data**: Ensure data exists before visualization
- **Graceful failures**: Handle empty results appropriately
```python
# Safely handle empty results
words, counts = zip(*frequencies) if frequencies else ([], [])
if words:
    plt.bar(words, counts)
else:
    print("No tokens found to display")
```

## Common Tasks

### Starting a New Analysis Notebook
1. Create markdown cell with title and description
2. Add import cell with sys.path.append('..')
3. Import required modules from src/
4. Add %matplotlib inline
5. Create sections with markdown headers
6. Add experimentation cell at end

### Exploring New Visualizations
1. Import visualization function from src.visualization
2. Prepare sample data (tokens, embeddings, etc.)
3. Create matplotlib figure with appropriate size
4. Call visualization function
5. Add descriptive labels and title
6. Document findings in markdown cell

### Testing Module Functions
1. Import function from appropriate src module
2. Create simple test case with known output
3. Run function and display results
4. Visualize results if applicable
5. Document observations

### Creating Educational Examples
1. Choose focused concept (tokenization, temperature, etc.)
2. Use simple, clear example text
3. Show step-by-step process
4. Visualize each stage
5. Add explanatory markdown between steps
6. Provide template for user experimentation

## Error Handling

### Import Errors
- **Issue**: Cannot import from src modules
- **Solution**: Verify sys.path.append('..') is in first code cell
- **Check**: Notebook is in notebooks/ directory, src/ is at parent level
```python
import sys
sys.path.append('..')  # Must be before src imports

try:
    from src.tokenization import tokenize
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from notebooks/ directory")
```

### Empty Data Handling
- **Issue**: Visualization functions fail with empty data
- **Solution**: Check data before plotting
```python
if len(tokens) > 0:
    frequencies = get_token_frequencies(tokens)
    # ... create plot
else:
    print("No tokens to analyze. Please provide text input.")
```

### Display Issues
- **Issue**: Plots not showing inline
- **Solution**: Ensure %matplotlib inline is executed
- **Alternative**: Use plt.show() explicitly after each plot

## Testing Guidelines

### Notebook Validation
- **Run all cells**: Use "Run All" to ensure notebook executes sequentially
- **Check outputs**: Verify all visualizations appear correctly
- **Test with fresh kernel**: Restart kernel and run all to test reproducibility

### Experimentation Safety
- **Save before experimenting**: Create checkpoint before major changes
- **Use separate cells**: Don't modify working cells, create new ones
- **Document findings**: Add markdown cells with observations

## Key Dependencies

### sys and path manipulation
- Required for importing from parent src/ directory
```python
import sys
sys.path.append('..')
```

### numpy (>=1.24.0)
- Numerical operations and array handling
- Used in temperature analysis and embeddings
```python
import numpy as np
temps = np.linspace(0.1, 2.0, 20)
```

### matplotlib (>=3.7.0)
- All visualization and plotting
- Interactive inline display with %matplotlib inline
```python
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(10, 5))
plt.plot(x, y)
plt.show()
```

### src modules
- tokenization: Text processing (tokenize, get_token_frequencies)
- embeddings: Vector operations (create_similarity_matrix, create_word_embedding)
- sampling: Temperature analysis (temperature_entropy_analysis, softmax)
- visualization: Plotting functions (plot_token_frequencies, plot_similarity_heatmap)

## Examples

### Complete Notebook Section
```markdown
## 2. Temperature vs Entropy
Temperature controls the randomness of LLM outputs. 
Higher temperature = more random = higher entropy.
```

```python
# Generate temperature-entropy relationship
temps, entropies = temperature_entropy_analysis(vocab_size=30)

# Create visualization
plt.figure(figsize=(8, 5))
plt.plot(temps, entropies, marker='o', linewidth=2, color='steelblue')
plt.xlabel('Temperature', fontsize=12)
plt.ylabel('Entropy (bits)', fontsize=12)
plt.title('Temperature vs Output Entropy', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

```markdown
**Observation**: As temperature increases, the output becomes more random,
resulting in higher entropy values. At very low temperatures (< 0.5), 
the model becomes nearly deterministic.
```

### Token Frequency Analysis
```python
# Analyze token frequencies in sample text
text = "Large language models generate text token by token using neural networks."
tokens = tokenize(text)

# Get top 5 most frequent tokens
frequencies = get_token_frequencies(tokens, top_k=5)
words, counts = zip(*frequencies) if frequencies else ([], [])

# Visualize
plt.figure(figsize=(10, 5))
plt.bar(words, counts, color='coral', edgecolor='darkred', linewidth=1.5)
plt.xlabel('Tokens', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Token Frequency Distribution', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print(f"Total tokens: {len(tokens)}")
print(f"Unique tokens: {len(set(tokens))}")
```

### Interactive Word Similarity Exploration
```python
# Define words to compare
words = ['model', 'neural', 'network', 'token', 'embedding', 'attention']

# Create similarity matrix with fixed dimensions for reproducibility
sim_matrix = create_similarity_matrix(words, dimensions=32)

# Visualize with heatmap
plt.figure(figsize=(8, 6))
im = plt.imshow(sim_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, label='Cosine Similarity')
plt.xticks(range(len(words)), words, rotation=45, ha='right')
plt.yticks(range(len(words)), words)
plt.title('Word Similarity Heatmap', fontsize=14, fontweight='bold')

# Add value annotations
for i in range(len(words)):
    for j in range(len(words)):
        text = plt.text(j, i, f'{sim_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

plt.tight_layout()
plt.show()
```

### User Experimentation Template
```python
# 🧪 Try Your Own Text!
# Replace the text below and run this cell to see the analysis

custom_text = "Your text here..."

# Tokenize and analyze
custom_tokens = tokenize(custom_text)
frequencies = get_token_frequencies(custom_tokens, top_k=10)

# Display results
print(f"📊 Total tokens: {len(custom_tokens)}")
print(f"📚 Unique tokens: {len(set(custom_tokens))}")
print(f"\n🔝 Top tokens:")
for word, count in frequencies:
    print(f"  {word}: {count}")

# Visualize
if frequencies:
    words, counts = zip(*frequencies)
    plt.figure(figsize=(10, 5))
    plt.bar(words, counts)
    plt.xlabel('Tokens')
    plt.ylabel('Frequency')
    plt.title('Your Text - Token Frequency Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
```
