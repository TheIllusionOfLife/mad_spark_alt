# Mad Spark Alt Commands Reference

This document lists all available commands in the Mad Spark Alt system.

## QADI Analysis Commands

These are the main commands for running QADI (Question-Abduction-Deduction-Induction) analysis.

### 1. `qadi_simple.py` - **RECOMMENDED DEFAULT**
The simplified QADI with clearer Phase 1 that just identifies the user's question.
```bash
uv run python qadi_simple.py "Your question here"
uv run python qadi_simple.py "Your question" --temperature 0.5  # More focused
uv run python qadi_simple.py "Your question" --verbose  # Show scores
```

### 2. `qadi_hypothesis.py` - Original Hypothesis-Driven
The original QADI implementation that tries to extract a "core question".
```bash
uv run python qadi_hypothesis.py "Your question here"
uv run python qadi_hypothesis.py "Your question" --temperature 1.2  # More creative
uv run python qadi_hypothesis.py "Your question" --verbose
```

### 3. `qadi_simple_multi.py` - Multi-Agent Analysis
Runs QADI with multiple LLM agents for each phase.
```bash
uv run python qadi_simple_multi.py "Your question here"
uv run python qadi_simple_multi.py --type=business "Business question"
uv run python qadi_simple_multi.py --concrete "Build something"
```

### 4. `qadi_multi_perspective.py` - Multi-Perspective Analysis
Analyzes questions from multiple perspectives (personal, community, systemic).
```bash
uv run python qadi_multi_perspective.py "Your question here"
uv run python qadi_multi_perspective.py --perspectives=3 "Your question"
uv run python qadi_multi_perspective.py --temperature 0.7 "Your question"
```

### 5. `qadi.py` - Basic Single-Prompt Version
Simple implementation using a single LLM call with Google API.
```bash
uv run python qadi.py "Your question here"
```

## CLI Commands (via mad-spark)

The installed package provides a `mad-spark` CLI:

```bash
# List available evaluators
uv run mad-spark list-evaluators

# Evaluate single text
uv run mad-spark evaluate "The AI dreamed of electric sheep"

# Help
uv run mad-spark --help
```

## Example Scripts

Located in the `examples/` directory:

### Individual Method Demos
```bash
# Test questioning method
uv run python examples/llm_questioning_demo.py

# Test abduction (hypothesis generation)
uv run python examples/llm_abductive_demo.py

# Test deduction (logical reasoning)
uv run python examples/llm_deductive_demo.py

# Test induction (pattern finding)
uv run python examples/llm_inductive_demo.py
```

### Full System Demos
```bash
# QADI cycle demo
uv run python examples/qadi_demo.py

# LLM showcase with all methods
uv run python examples/llm_showcase_demo.py

# Evolution demo (genetic algorithm)
uv run python examples/evolution_demo.py
```

## Common Options

### Temperature Control
Most QADI commands support temperature control for creativity:
- `--temperature 0.0` - Very focused, deterministic
- `--temperature 0.5` - Balanced (default for most)
- `--temperature 0.8` - Default for hypothesis generation
- `--temperature 1.2` - More creative
- `--temperature 2.0` - Maximum creativity

### Verbose Mode
Add `--verbose` or `-v` to see detailed scores and analysis.

## Environment Setup

All commands require Google API key:
```bash
export GOOGLE_API_KEY="your-key-here"
```

Optional for other providers:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

## Quick Start Recommendations

1. **For most users**: Start with `qadi_simple.py` - it's the clearest and most user-friendly
2. **For creative exploration**: Use higher temperature (1.0-1.5)
3. **For focused analysis**: Use lower temperature (0.3-0.7)
4. **For business context**: Use `qadi_simple_multi.py --type=business`
5. **For implementation planning**: Use `qadi_simple_multi.py --concrete`

## Key Differences

- **qadi_simple.py**: Simplified Phase 1, clearest output
- **qadi_hypothesis.py**: Original version, tries to find "core question"
- **qadi_simple_multi.py**: Multiple agents, type detection
- **qadi_multi_perspective.py**: Analyzes from 3 perspectives
- **qadi.py**: Single LLM call, fastest but less structured