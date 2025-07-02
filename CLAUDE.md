# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mad Spark Alt is an AI Creativity Evaluation System implementing a **Hybrid Multi-layer Evaluation Framework**. The system evaluates AI model creativity through three complementary layers:

1. **Quantitative Layer** - Automated metrics for diversity, quality analysis
2. **LLM Judge Layer** - AI models evaluate creativity using structured prompts  
3. **Human Evaluation Layer** - Expert/user assessment (in development)

## Commands

### Development Environment
```bash
# Install dependencies (preferred)
uv sync

# Install with dev dependencies
uv sync --dev

# Alternative with pip
pip install -e .
```

### Testing & Quality
```bash
# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/unit/test_llm_judges.py

# Type checking
uv run mypy src/

# Code formatting
uv run black src/ tests/

# Import sorting
uv run isort src/ tests/
```

### CLI Usage
```bash
# Main CLI entry point
mad-spark --help

# List available evaluators
mad-spark list-evaluators

# List LLM judges
mad-spark list-judges

# Test LLM connections
mad-spark test-judges

# Evaluate single text
mad-spark evaluate "creative text" --model gpt-4

# LLM judge evaluation
mad-spark evaluate "text" --llm-judge gpt-4

# Multi-judge jury 
mad-spark evaluate "text" --jury "gpt-4,claude-3-sonnet,gemini-pro"

# Pre-configured jury budgets
mad-spark evaluate "text" --jury-budget balanced
```

### Environment Variables
LLM Judge functionality requires API keys:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key" 
export GOOGLE_API_KEY="your-google-key"
```

## Architecture

### Core Structure
```
src/mad_spark_alt/
├── core/                    # Main evaluation engine
│   ├── evaluator.py        # CreativityEvaluator orchestrator
│   ├── interfaces.py       # Abstract base classes & data models
│   └── registry.py         # Plugin registration system
├── layers/                 # Evaluation layer implementations
│   ├── quantitative/       # Layer 1: Automated metrics
│   │   ├── diversity.py    # Lexical/semantic diversity metrics
│   │   └── quality.py      # Fluency, grammar, readability
│   ├── llm_judges/         # Layer 2: AI evaluators
│   │   ├── base.py         # CreativityLLMJudge base class
│   │   ├── jury.py         # CreativityJury multi-judge system
│   │   ├── prompts.py      # Evaluation prompt templates
│   │   ├── providers.py    # OpenAI/Anthropic/Google clients
│   │   └── config.py       # Model configurations & budgets
│   └── human_eval/         # Layer 3: Human assessment (placeholder)
├── models/                 # Data models & schemas
├── storage/                # Persistence layer
└── cli.py                  # Command-line interface
```

### Key Classes

**Core Interfaces** (`core/interfaces.py`):
- `EvaluatorInterface`: Abstract base for all evaluators
- `EvaluationRequest`: Input data structure  
- `EvaluationResult`: Output data structure
- `ModelOutput`: Represents AI-generated content to evaluate
- `EvaluationLayer`: Enum (QUANTITATIVE, LLM_JUDGE, HUMAN)
- `OutputType`: Enum (TEXT, CODE, IMAGE, STRUCTURED)

**Main Orchestrator** (`core/evaluator.py`):
- `CreativityEvaluator`: Coordinates evaluation across all layers
- Handles async execution, result aggregation, scoring

**Plugin System** (`core/registry.py`):
- Dynamic evaluator registration
- Automatic discovery of implementations

### Layer Implementations

**Quantitative Layer**:
- `DiversityEvaluator`: Distinct-N, semantic uniqueness, lexical diversity
- `QualityEvaluator`: Fluency, grammar, readability, coherence

**LLM Judge Layer**:
- `CreativityLLMJudge`: Single AI model evaluation
- `CreativityJury`: Multi-judge consensus with disagreement detection
- Supports GPT-4, Claude-3, Gemini models
- Cost tracking and budget management

## Development Patterns

### Adding New Evaluators
Implement `EvaluatorInterface` and register:

```python
from mad_spark_alt.core import EvaluatorInterface, register_evaluator

class MyEvaluator(EvaluatorInterface):
    @property
    def name(self) -> str:
        return "my_evaluator"
    
    @property
    def layer(self) -> EvaluationLayer:
        return EvaluationLayer.QUANTITATIVE
    
    async def evaluate(self, request: EvaluationRequest) -> List[EvaluationResult]:
        # Implementation here
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        return True

register_evaluator(MyEvaluator)
```

### Async Pattern
All evaluators use async/await for I/O operations:
- LLM API calls are concurrent where possible
- Batch processing for efficiency
- Proper error handling and timeouts

### Data Flow
1. `EvaluationRequest` created with `ModelOutput` list
2. `CreativityEvaluator.evaluate()` routes to appropriate layers
3. Each evaluator returns `EvaluationResult` list  
4. Results aggregated into `EvaluationSummary`
5. Overall creativity score calculated

### Error Handling
- Graceful fallbacks for missing API keys
- Individual evaluator failures don't crash entire evaluation
- Structured error responses with context

## Testing Strategy

### Test Structure
```
tests/
├── unit/                   # Unit tests for individual components
│   ├── test_core.py       # Core functionality tests
│   └── test_llm_judges.py # LLM judge specific tests
└── integration/           # End-to-end integration tests
```

### Mock Strategy
- Mock external API calls to avoid costs during testing
- Test data includes various creativity dimensions
- Separate tests for different output types (text, code)

### Test Execution
Run tests without external dependencies by default. Use environment variables to enable real API testing during development.

## Important Implementation Notes

### Cost Management
LLM judges track token usage and costs. Budget-aware jury configurations prevent runaway expenses.

### Multi-Judge Consensus
The jury system uses voting mechanisms to handle disagreements between AI judges, improving evaluation reliability.

### Extensibility
The plugin architecture allows easy addition of new evaluation methods without modifying core code.

### Performance
Async execution enables parallel evaluation across multiple AI services, significantly reducing total evaluation time.