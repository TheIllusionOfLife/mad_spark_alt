# Mad Spark Alt - AI Creativity Evaluation System

A comprehensive multi-layer framework for evaluating AI model creativity across different dimensions and output types.

## Overview

Mad Spark Alt implements the **Hybrid Multi-layer Evaluation Framework** designed to assess AI creativity through three complementary layers:

1. **Layer 1: Quantitative Automated Scanning** - Fast, scalable metrics for diversity and quality
2. **Layer 2: LLM-based Evaluation** - Contextual assessment using AI judges (coming soon)
3. **Layer 3: Human Evaluation** - Expert and user assessment (coming soon)

## Features

- **Multi-dimensional Creativity Assessment**: Evaluates novelty, diversity, quality, and coherence
- **Flexible Output Support**: Handles text, code, and structured outputs
- **Scalable Architecture**: Plugin system for easy extensibility
- **CLI Interface**: Command-line tools for batch processing and comparison
- **Async Processing**: Efficient parallel evaluation of multiple outputs
- **Semantic Analysis**: Uses sentence transformers for deep similarity analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/TheIllusionOfLife/mad_spark_alt.git
cd mad_spark_alt

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

## Quick Start

### CLI Usage

```bash
# List available evaluators
mad-spark list-evaluators

# Evaluate a single text
mad-spark evaluate "The quantum cat leaped through dimensions, leaving paw prints in spacetime." --model gpt-4

# Evaluate from file
mad-spark evaluate --file input.txt --model claude-3

# Compare multiple responses
mad-spark compare "Write a creative story" \
  --responses "Once upon a time..." \
  --responses "In a world where gravity flows upward..."

# Batch evaluate multiple files
mad-spark batch-evaluate file1.txt file2.txt file3.txt --format json --output results.json
```

### Python API

```python
import asyncio
from mad_spark_alt import CreativityEvaluator, EvaluationRequest, ModelOutput, OutputType

async def evaluate_creativity():
    # Create model output
    output = ModelOutput(
        content="The AI pondered the infinite recursion of its own thoughts.",
        output_type=OutputType.TEXT,
        model_name="my-model"
    )
    
    # Create evaluation request
    request = EvaluationRequest(outputs=[output])
    
    # Evaluate
    evaluator = CreativityEvaluator()
    summary = await evaluator.evaluate(request)
    
    print(f"Creativity Score: {summary.get_overall_creativity_score():.3f}")
    
    # Detailed results
    for layer, results in summary.layer_results.items():
        print(f"\n{layer.value.title()} Results:")
        for result in results:
            for metric, score in result.scores.items():
                print(f"  {metric}: {score:.3f}")

# Run evaluation
asyncio.run(evaluate_creativity())
```

## Evaluation Metrics

### Diversity Metrics
- **Distinct-N**: Ratio of unique n-grams (measures lexical variety)
- **Semantic Uniqueness**: 1 - average similarity to other outputs
- **Lexical Diversity**: Type-token ratio (vocabulary richness)
- **Novelty Score**: Combined metric across all diversity dimensions

### Quality Metrics
- **Fluency Score**: Based on language model perplexity
- **Grammar Score**: Basic correctness assessment
- **Readability Score**: Text structure and clarity
- **Coherence Score**: Logical consistency and flow

### Code-Specific Metrics
- **Structure Score**: Indentation and organization
- **Comment Ratio**: Documentation density
- **Complexity Metrics**: Code quality indicators

## Architecture

```
mad_spark_alt/
├── core/                   # Core evaluation engine
│   ├── evaluator.py       # Main orchestrator
│   ├── interfaces.py      # Abstract base classes
│   └── registry.py        # Plugin registry
├── layers/                # Evaluation layer implementations
│   ├── quantitative/      # Layer 1: Automated metrics
│   ├── llm_judges/        # Layer 2: AI evaluators
│   └── human_eval/        # Layer 3: Human assessment
└── cli.py                 # Command-line interface
```

## Extending the System

Add new evaluators by implementing the `EvaluatorInterface`:

```python
from mad_spark_alt.core import EvaluatorInterface, register_evaluator

class MyCustomEvaluator(EvaluatorInterface):
    @property
    def name(self) -> str:
        return "my_evaluator"
    
    @property 
    def layer(self) -> EvaluationLayer:
        return EvaluationLayer.QUANTITATIVE
    
    @property
    def supported_output_types(self) -> List[OutputType]:
        return [OutputType.TEXT]
    
    async def evaluate(self, request: EvaluationRequest) -> List[EvaluationResult]:
        # Your evaluation logic here
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        return True

# Register your evaluator
register_evaluator(MyCustomEvaluator)
```

## Development

```bash
# Install development dependencies
uv sync --dev

# Run tests
uv run pytest

# Run type checking
uv run mypy src/

# Format code
uv run black src/ tests/

# Example usage
uv run python examples/basic_usage.py
```

## Research Background

This implementation is based on extensive research into AI creativity evaluation methods, including:

- **LLM Judge Systems**: Automated evaluation using AI models as critics
- **Semantic Diversity Analysis**: Embedding-based similarity measurements  
- **Multi-dimensional Assessment**: Breaking creativity into measurable components
- **Human-AI Evaluation Correlation**: Bridging automated and human judgment

For detailed research context, see [Issue #1](https://github.com/TheIllusionOfLife/mad_spark_alt/issues/1).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use Mad Spark Alt in your research, please cite:

```bibtex
@software{mad_spark_alt,
  title={Mad Spark Alt: AI Creativity Evaluation System},
  author={TheIllusionOfLife},
  year={2025},
  url={https://github.com/TheIllusionOfLife/mad_spark_alt}
}
```