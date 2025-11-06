# CLI Usage Guide

## Overview

Mad Spark Alt provides a comprehensive command-line interface for hypothesis-driven QADI analysis, idea evolution, and creativity evaluation. This guide covers all available commands and usage patterns.

## Installation and Setup

```bash
# Install the package
pip install -e .

# Or with uv (recommended)
uv sync

# Verify installation
uv run mad-spark --help  # Note: use 'uv run' prefix with uv installation
```

## Environment Setup

### Required Environment Variables

For QADI multi-agent analysis and LLM functionality, set up API keys:

```bash
# Google Gemini API (RECOMMENDED - primary LLM provider)
export GOOGLE_API_KEY="your-google-api-key"

# Optional: OpenAI API (alternative LLM provider)
export OPENAI_API_KEY="your-openai-api-key"

# Optional: Anthropic Claude API (alternative LLM provider)  
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Alternative: Create .env file in project root
echo "GOOGLE_API_KEY=your-key-here" > .env
```

**Note**: At least one LLM API key is required for the QADI system to function. Google API is recommended for best results and lowest cost.

### Optional Configuration

```bash
# Set custom config directory
export MAD_SPARK_CONFIG_DIR="~/.config/mad_spark_alt"

# Enable detailed logging
export MAD_SPARK_LOG_LEVEL="DEBUG"
```

## Basic Commands

### Global Options

All commands support these global options:

```bash
uv run mad-spark --help                    # Show help
uv run mad-spark --verbose                 # Enable verbose output
```

### Core Commands

```bash
# Main help
uv run mad-spark --help

# List available evaluators
uv run mad-spark list-evaluators
```

## 🧬 Evolution Command (MOST POWERFUL FEATURE)

The `evolve` command combines QADI hypothesis generation with genetic algorithms to find optimal solutions:

### Basic Evolution Usage

```bash
# Basic evolution
uv run mad-spark evolve "How can we reduce food waste?"

# With custom context
uv run mad-spark evolve "Improve customer service" --context "E-commerce platform"

# With custom parameters
uv run mad-spark evolve "Climate solutions" --generations 2

# Save results to file
uv run mad-spark evolve "New product ideas" --output results.json
```

### Advanced Evolution Options

```bash
# Custom evolution parameters
uv run mad-spark evolve "Business strategies" \
  --generations 5 \          # Number of evolution generations (default: 3)
  --population 20 \          # Population size (default: 12)
  --temperature 1.5          # Creativity level 0.0-2.0 (default: 0.8)

# Full example with all options
uv run mad-spark evolve "How to increase team productivity" \
  --context "Remote software development team" \
  --generations 10 \
  --population 30 \
  --temperature 1.2 \
  --output evolution_results.json
```

### Temperature Control

The `--temperature` flag controls creativity in hypothesis generation:
- `0.0-0.5`: Conservative, practical ideas
- `0.6-1.0`: Balanced creativity (default: 0.8)
- `1.1-2.0`: Highly creative, unconventional ideas

### Evolution Process

1. **QADI Analysis**: Generates initial hypotheses
2. **Fitness Evaluation**: Scores on 5 criteria (novelty, impact, cost, feasibility, risks)
3. **Selection**: Best ideas selected for breeding
4. **Crossover**: Combines elements from parent ideas
5. **Mutation**: Introduces variations
6. **Repeat**: Continues for specified generations

### Example Output

```
🧬 Evolution Pipeline
Problem: How can we reduce plastic waste in oceans?
Generations: 5 | Population: 12 | Temperature: 0.8

✅ Generated 3 initial hypotheses
💰 LLM Cost: $0.0234

✅ Evolution completed in 45.2s

🏆 Top Evolved Ideas:
1. Autonomous ocean drones with ML-powered plastic detection... (Fitness: 0.892)
2. Blockchain-tracked plastic credits incentivizing cleanup... (Fitness: 0.847)
3. Bioengineered bacteria that safely decompose ocean plastic... (Fitness: 0.823)

📊 Results:
• Fitness improvement: 47.3%
• Ideas evaluated: 60
• Best from generation: 4

💾 Cache Performance:
• Hit rate: 65%
• LLM calls saved: 39
```

## Creativity Evaluation Commands

### Basic Evaluation

```bash
# Evaluate a single text
mad-spark evaluate "The quantum cat leaped through dimensions, leaving paw prints in spacetime."

# Evaluate text with specific model context
mad-spark evaluate "Creative text here" --model gpt-4 --context "Creative writing task"

# Evaluate from file
mad-spark evaluate --file input.txt

# Evaluate from stdin
echo "Creative content" | mad-spark evaluate --stdin
```

### Advanced Evaluation Options

```bash
# Specify output type
mad-spark evaluate "Python code here" --output-type code

# Custom evaluation layers
mad-spark evaluate "text" --layers quantitative,llm_judge

# Save results to file
mad-spark evaluate "text" --output results.json

# Pretty print results
mad-spark evaluate "text" --format table
mad-spark evaluate "text" --format json --pretty
```

### LLM Judge Evaluation

```bash
# Single LLM judge
mad-spark evaluate "Creative text" --llm-judge gpt-4
mad-spark evaluate "Creative text" --llm-judge claude-3-sonnet
mad-spark evaluate "Creative text" --llm-judge gemini-pro

# Multiple judge jury
mad-spark evaluate "text" --jury "gpt-4,claude-3-sonnet,gemini-pro"

# Pre-configured jury budgets
mad-spark evaluate "text" --jury-budget economy      # Low cost judges
mad-spark evaluate "text" --jury-budget balanced     # Mix of judges  
mad-spark evaluate "text" --jury-budget premium      # High-end judges

# Custom jury configuration
mad-spark evaluate "text" --jury-config jury_config.json
```

### Batch Evaluation

```bash
# Evaluate multiple files
mad-spark batch-evaluate file1.txt file2.txt file3.txt

# Batch with custom output
mad-spark batch-evaluate *.txt --output batch_results.json --format json

# Batch with table output
mad-spark batch-evaluate *.txt --format table

# Batch with progress tracking
mad-spark batch-evaluate large_file1.txt large_file2.txt --progress
```

### Comparison and Analysis

```bash
# Compare multiple outputs
mad-spark compare "output1" "output2" "output3"

# Compare files
mad-spark compare --files file1.txt file2.txt file3.txt

# Compare with reference
mad-spark compare "test_output" --reference "gold_standard.txt"

# Diversity analysis
mad-spark diversity-analysis file1.txt file2.txt file3.txt --metric all
```

## QADI System Commands

### QADI Analysis (Recommended)

Use the CLI for QADI multi-agent analysis:

```bash
# Basic QADI analysis
uv run mad-spark qadi "How can we improve urban sustainability?"

# View help and all options
uv run mad-spark qadi --help

# View all CLI options
uv run mad-spark --help
```

### Using the Python API

For programmatic access, use `SimpleQADIOrchestrator`:

```python
import asyncio
import os
from mad_spark_alt.core import SimpleQADIOrchestrator, setup_llm_providers

async def run_qadi():
    # 1. Setup LLM provider (requires API key)
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    await setup_llm_providers(google_api_key=google_api_key)

    # 2. Create orchestrator (uses global llm_manager)
    orchestrator = SimpleQADIOrchestrator(num_hypotheses=3)

    # 3. Run QADI cycle
    result = await orchestrator.run_qadi_cycle("Your question here")
    print(result.hypotheses)

# Run the async function
asyncio.run(run_qadi())
```

### Legacy Scripts (Deprecated)

```bash
# Quick single-prompt QADI
uv run python qadi.py "Your question here"

# Interactive QADI demonstration
uv run python examples/qadi_demo.py

# Basic usage examples
uv run python examples/basic_usage.py
```

### Genetic Evolution CLI

Use the built-in evolution command for idea refinement:

```bash
# Basic evolution (requires GOOGLE_API_KEY)
uv run mad-spark evolve "How can we reduce food waste?"

# Evolution with context and custom parameters
uv run mad-spark evolve "Improve remote work" \
  --context "Focus on team collaboration" \
  --generations 5 \
  --population 15

# Evolution with minimum parameters
uv run mad-spark evolve "Climate solutions" --generations 2 --population 2

# Save evolution results
uv run mad-spark evolve "Innovation challenge" \
  --output evolution_results.json \
  --generations 3 \
  --population 12

# Evolution demo examples
uv run python examples/evolution_demo.py
```

### Agent Management

```bash
# List available thinking agents
mad-spark list-agents

# Show agent details
mad-spark agent-info QuestioningAgent
mad-spark agent-info --all

# Test individual agent
mad-spark test-agent QuestioningAgent "Test problem statement"

# Test all agents
mad-spark test-agents "Test problem for all agents"
```

### QADI Cycle Execution

```bash
# Run complete QADI cycle
mad-spark qadi-cycle "How can we reduce plastic waste in urban environments?"

# QADI with custom configuration
mad-spark qadi-cycle "Problem statement" \
  --config qadi_config.json \
  --max-ideas 5 \
  --require-reasoning \
  --context "Additional context information"

# QADI with specific agents only
mad-spark qadi-cycle "Problem" --agents questioning,abduction,deduction

# Save QADI results
mad-spark qadi-cycle "Problem" --output qadi_results.json --format json
```

### Parallel Generation

```bash
# Run specific thinking methods in parallel
mad-spark parallel-generate "Innovation challenge" \
  --methods questioning,abduction \
  --max-ideas 3 \
  --context "Technology startup context"

# Parallel generation with custom config
mad-spark parallel-generate "Problem" \
  --methods all \
  --config parallel_config.json \
  --output parallel_results.json
```

## Configuration Files

### Basic Configuration

Create `.mad_spark_config.json` in your project or home directory:

```json
{
  "default_model": "gpt-4",
  "default_max_ideas": 5,
  "default_require_reasoning": true,
  "evaluation": {
    "default_layers": ["quantitative", "llm_judge"],
    "default_output_format": "table"
  },
  "qadi": {
    "default_agents": ["questioning", "abduction", "deduction", "induction"],
    "default_creativity_level": "balanced"
  },
  "llm_judges": {
    "default_jury": ["gpt-4", "claude-3-sonnet"],
    "budget_presets": {
      "economy": ["gpt-3.5-turbo"],
      "balanced": ["gpt-4", "claude-3-sonnet"],
      "premium": ["gpt-4", "claude-3-opus", "gemini-pro"]
    }
  }
}
```

### QADI Configuration

Create `qadi_config.json` for QADI-specific settings:

```json
{
  "cycle_config": {
    "max_ideas_per_method": 3,
    "require_reasoning": true,
    "creativity_level": "balanced"
  },
  "agent_configs": {
    "questioning": {
      "question_types": ["clarifying", "alternative", "challenging"],
      "focus_areas": ["assumptions", "constraints", "stakeholders"]
    },
    "abduction": {
      "hypothesis_types": ["causal", "analogical", "pattern"],
      "creativity_level": "balanced",
      "use_analogies": true
    },
    "deduction": {
      "reasoning_modes": ["logical", "constraint", "implication"],
      "validation_depth": "thorough"
    },
    "induction": {
      "synthesis_methods": ["pattern", "principle", "insight"],
      "pattern_depth": "deep"
    }
  }
}
```

### Jury Configuration

Create `jury_config.json` for custom LLM judge setups:

```json
{
  "judges": [
    {
      "model": "gpt-4",
      "weight": 0.4,
      "specialization": "general_creativity"
    },
    {
      "model": "claude-3-sonnet", 
      "weight": 0.4,
      "specialization": "logical_reasoning"
    },
    {
      "model": "gemini-pro",
      "weight": 0.2,
      "specialization": "novelty_assessment"
    }
  ],
  "voting_strategy": "weighted_average",
  "disagreement_threshold": 0.3,
  "require_consensus": false
}
```

## Output Formats

### JSON Format

```bash
mad-spark evaluate "text" --format json --output results.json
```

```json
{
  "overall_creativity_score": 0.753,
  "execution_time": 2.34,
  "layer_results": {
    "quantitative": [
      {
        "evaluator_name": "diversity_evaluator",
        "scores": {
          "novelty_score": 0.72,
          "semantic_uniqueness": 0.85,
          "lexical_diversity": 0.64
        }
      }
    ]
  },
  "summary": {
    "strengths": ["High semantic uniqueness", "Good lexical variety"],
    "areas_for_improvement": ["Could increase novelty"],
    "overall_assessment": "Above average creativity"
  }
}
```

### Table Format

```bash
mad-spark evaluate "text" --format table
```

```
┌─────────────────────┬─────────┬─────────────────────────────────┐
│ Evaluator           │ Score   │ Details                         │
├─────────────────────┼─────────┼─────────────────────────────────┤
│ diversity_evaluator │ 0.753   │ novelty: 0.72, uniqueness: 0.85│
│ quality_evaluator   │ 0.821   │ fluency: 0.89, coherence: 0.75 │
├─────────────────────┼─────────┼─────────────────────────────────┤
│ Overall             │ 0.787   │ Above average creativity        │
└─────────────────────┴─────────┴─────────────────────────────────┘
```

### CSV Format

```bash
mad-spark batch-evaluate *.txt --format csv --output results.csv
```

## Advanced Usage

### Scripting and Automation

```bash
#!/bin/bash
# Evaluation pipeline script

echo "Running creativity evaluation pipeline..."

# Batch evaluate all text files
mad-spark batch-evaluate inputs/*.txt \
  --format json \
  --output evaluation_results.json \
  --llm-judge gpt-4

# Run QADI analysis on top-performing texts
for file in inputs/*.txt; do
  score=$(mad-spark evaluate "$file" --format json | jq '.overall_creativity_score')
  if (( $(echo "$score > 0.8" | bc -l) )); then
    echo "Running QADI analysis on high-scoring file: $file"
    mad-spark qadi-cycle "$(cat $file)" \
      --output "qadi_$(basename $file .txt).json"
  fi
done

echo "Pipeline complete!"
```

### Integration with Other Tools

```bash
# Pipe from other creativity tools
creativity-generator --topic "AI" | mad-spark evaluate --stdin

# Chain with text processing
mad-spark evaluate "$(cat creative_text.txt | sed 's/old/new/g')"

# Use with version control
git diff HEAD~1 --name-only "*.md" | xargs mad-spark batch-evaluate
```

### Performance Optimization

```bash
# Parallel evaluation for large batches
mad-spark batch-evaluate large_dataset/*.txt \
  --parallel \
  --max-workers 4 \
  --cache-enabled

# Memory-efficient processing
mad-spark batch-evaluate huge_files/*.txt \
  --streaming \
  --chunk-size 1000

# Fast evaluation with minimal layers
mad-spark evaluate "text" --layers quantitative --fast-mode
```

## Testing and Debugging

### System Diagnostics

```bash
# Test system health
mad-spark doctor

# Test all components
mad-spark test-system

# Test LLM connectivity
mad-spark test-llm-judges

# Test QADI agents
mad-spark test-qadi-system
```

### Debug Mode

```bash
# Enable debug output
mad-spark --verbose evaluate "text"

# Detailed agent execution
mad-spark qadi-cycle "problem" --debug --trace-execution

# Performance profiling
mad-spark evaluate "text" --profile --timing-details
```

### Validation Commands

```bash
# Validate configuration
mad-spark validate-config config.json

# Validate input files
mad-spark validate-input file1.txt file2.txt

# Validate QADI setup
mad-spark validate-qadi-setup
```

## Error Handling and Troubleshooting

### Common Issues

```bash
# API key issues
mad-spark test-llm-judges  # Check API connectivity

# Permission issues
mad-spark doctor --check-permissions

# Configuration issues
mad-spark validate-config --fix-common-issues

# Missing dependencies
mad-spark doctor --check-dependencies
```

### Recovery Commands

```bash
# Reset configuration to defaults
mad-spark reset-config

# Clear caches
mad-spark clear-cache

# Rebuild agent registry
mad-spark rebuild-registry
```

## Examples and Use Cases

### Academic Research

```bash
# Evaluate research paper abstracts
mad-spark batch-evaluate abstracts/*.txt \
  --context "Academic research evaluation" \
  --llm-judge claude-3-sonnet \
  --output academic_results.json

# Generate research questions
mad-spark qadi-cycle "How can AI improve scientific discovery?" \
  --agents questioning \
  --max-ideas 10 \
  --context "Academic research context"
```

### Creative Writing

```bash
# Evaluate story creativity
mad-spark evaluate "$(cat story.txt)" \
  --context "Creative fiction evaluation" \
  --jury "gpt-4,claude-3-sonnet" \
  --format table

# Generate story ideas
mad-spark qadi-cycle "Create an engaging science fiction story" \
  --agents questioning,abduction \
  --creativity-level high
```

### Business Innovation

```bash
# Evaluate business proposals
mad-spark batch-evaluate proposals/*.txt \
  --context "Business innovation assessment" \
  --output business_evaluation.json

# Generate business solutions
mad-spark qadi-cycle "How can we improve customer retention?" \
  --context "SaaS business model" \
  --all-agents \
  --output retention_ideas.json
```

### Educational Applications

```bash
# Evaluate student creative writing
mad-spark batch-evaluate student_essays/*.txt \
  --format csv \
  --output student_creativity_scores.csv

# Generate learning questions
mad-spark qadi-cycle "How can we make learning more engaging?" \
  --agents questioning \
  --context "Educational technology"
```

This comprehensive CLI guide covers all aspects of using Mad Spark Alt from the command line, enabling users to leverage both the creativity evaluation and QADI idea generation capabilities effectively.