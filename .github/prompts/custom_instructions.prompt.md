---
description: Generate custom instructions for workspace modules and folders
mode: agent
---

# Generate Custom Instructions for Workspace Modules

Please analyze my workspace structure and generate tailored custom instructions for different types of modules and folders. Create specific instruction files that GitHub Copilot can use to provide contextually appropriate assistance.

## Requirements

Generate separate instruction files with the following naming pattern:
- `{folder-name}.instructions.md` (e.g., `tests.instructions.md`, `src.instructions.md`)
- Each file must include a YAML frontmatter header with `applyTo` field

**IMPORTANT**: Only generate instructions for folders that are explicitly provided as context in the request. Do NOT generate instructions for every folder in the repository automatically. Do not guess! Do not generate any instructions unless the folder is explicitly mentioned. If no specific folder is provided as context, ask for clarification about which folder(s) need instructions.

## Module Types to Cover

### 1. Test Folders (`tests/`, `test/`)
**File**: `tests.instructions.md`
- Header: `applyTo: tests/**`
- Focus on pytest patterns, test structure, and assertions
- Include testing best practices (arrange-act-assert, fixtures, mocking)
- Emphasize deterministic testing and edge case coverage

### 2. Source Code Folders (`src/`, `lib/`)
**File**: `src.instructions.md` (or match actual folder name)
- Header: `applyTo: src/**`
- Focus on clean code principles, type hints, and documentation
- Include import patterns and module organization
- Emphasize error handling and logging practices

### 3. Documentation Folders (`docs/`, `documentation/`)
**File**: `docs.instructions.md` (or match actual folder name)
- Header: `applyTo: docs/**`
- Focus on clear technical writing and code examples
- Include API documentation and user guides
- Emphasize accessibility and maintainability

### 4. Configuration Root/Folders (`config/`, root config files)
**File**: `config.instructions.md`
- Header: `applyTo: config/**` or `applyTo: ["*.toml", "*.yaml", "*.json"]`
- Focus on configuration management and validation
- Include dependency management and build systems
- Emphasize environment-specific settings

### 5. Notebooks Folders (`notebooks/`, `analysis/`)
**File**: `notebooks.instructions.md` (or match actual folder name)
- Header: `applyTo: notebooks/**`
- Focus on data exploration, visualization, and interactive development
- Include markdown documentation practices
- Emphasize reproducible research patterns

### 6. Scripts/CLI Folders (`scripts/`, `bin/`)
**File**: `scripts.instructions.md` (or match actual folder name)
- Header: `applyTo: scripts/**`
- Focus on argument parsing and user interface
- Include error handling and help text
- Emphasize user experience and validation

## Required File Format

Each instruction file MUST follow this exact structure:

```markdown
---
description: [Brief description of the module type and purpose]
applyTo: [folder-pattern]/**
---

# [Folder Name] Instructions

## Context
Brief description of when these instructions apply.

## Code Patterns
- Pattern 1: Description and example
- Pattern 2: Description and example

## Best Practices
- Practice 1: Explanation
- Practice 2: Explanation

## Common Tasks
- Task 1: How to approach it
- Task 2: How to approach it

## Error Handling
Guidelines for error handling specific to this module type.

## Testing Guidelines
How testing applies to this module type.

## Key Dependencies
Common libraries and their usage patterns.

## Examples
Brief code snippets demonstrating key concepts.
```

## ApplyTo Pattern Examples

- For tests folder: `applyTo: tests/**`
- For source folder: `applyTo: src/**`
- For multiple patterns: `applyTo: ["tests/**", "test/**"]`
- For file types: `applyTo: "**/*.py"`
- For root configs: `applyTo: ["*.toml", "*.yaml", "*.json", "*.cfg"]`

## Instructions

1. **Context Check**: Only process folders explicitly provided in the request context
2. **Ask for Clarification**: If no specific folder is mentioned, ask which folder(s) need instructions
3. **Analyze** the actual folder structure in the provided context
4. **Match** instruction file names exactly to existing folder names
5. **Include** proper YAML frontmatter with `applyTo` field
6. **Customize** content based on actual project patterns and dependencies
7. **Reference** existing code patterns found in the workspace
8. **Ensure** instructions are actionable and specific to each folder's purpose

Please generate comprehensive instruction files that will help GitHub Copilot provide contextually appropriate assistance for each folder type in my workspace.
