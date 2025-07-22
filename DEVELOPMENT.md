# Development Guide

This document provides comprehensive guidelines for developing, contributing to, and maintaining the Mad Spark Alt codebase.

## Table of Contents

- [Development Setup](#development-setup)
- [Architecture Overview](#architecture-overview)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Performance Considerations](#performance-considerations)
- [Contribution Workflow](#contribution-workflow)
- [Debugging and Troubleshooting](#debugging-and-troubleshooting)

## Development Setup

### Prerequisites

- Python 3.8+ (3.11+ recommended)
- [uv](https://github.com/astral-sh/uv) (recommended package manager)
- Git
- API keys for LLM providers (Google, OpenAI, Anthropic)

### Installation

```bash
# Clone and setup
git clone https://github.com/TheIllusionOfLife/mad_spark_alt.git
cd mad_spark_alt

# Install dependencies (preferred method)
uv sync --dev

# Install package in editable mode (required for CLI)
uv pip install -e .

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Development Environment Commands

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/mad_spark_alt --cov-report=html

# Type checking
uv run mypy src/

# Code formatting
uv run black src/ tests/

# Import sorting
uv run isort src/ tests/

# CLI commands (must use uv run prefix)
uv run mad-spark --help
uv run python qadi_hypothesis.py "Your question"
uv run mad-spark evolve "Your problem" --temperature 1.2
```

## Architecture Overview

### Core Design Principles

1. **Hypothesis-Driven**: True QADI methodology from consulting approach
2. **Simplicity First**: Removed unnecessary complexity (classifiers, adaptive prompts)
3. **Unified Evaluation**: Consistent 5-criteria scoring throughout
4. **Phase Optimization**: Each QADI phase uses optimal hyperparameters
5. **Async-First**: All operations designed for concurrency

### System Components

#### 1. QADI Core (`src/mad_spark_alt/core/`)

- **interfaces.py**: Core abstractions and protocols
- **simple_qadi_orchestrator.py**: Clean QADI implementation (NEW)
- **qadi_prompts.py**: Universal prompts for all phases (NEW)
- **unified_evaluator.py**: 5-criteria evaluation system (NEW)
- **llm_provider.py**: Unified LLM API interface
- **registry.py**: Component registration
- **smart_orchestrator.py**: Legacy multi-agent system (being phased out)

#### 2. Thinking Agents (`src/mad_spark_alt/agents/`)

- **questioning/**: Question formation and problem framing
- **abduction/**: Hypothesis generation and creative leaps
- **deduction/**: Logical reasoning and validation
- **induction/**: Pattern synthesis and rule formation

#### 3. Evaluation System (`src/mad_spark_alt/layers/`)

- **quantitative/**: Metrics-based evaluation (diversity, quality)
- **llm_judges/**: AI-powered creativity assessment
- **evaluation_utils.py**: Shared utilities for all evaluators

#### 4. Evolution System (`src/mad_spark_alt/evolution/`)

- **genetic_algorithm.py**: Main GA implementation
- **operators.py**: Crossover, mutation, selection operators
- **mutation_strategies.py**: Strategy pattern for mutations
- **fitness.py**: Unified fitness evaluation using 5-criteria system
- **cached_fitness.py**: Result caching (50-70% API reduction)
- **checkpointing.py**: Save/resume evolution state
- **llm_operators.py**: AI-powered genetic operators

### True QADI Methodology

The system implements the hypothesis-driven QADI approach from "Shin Logical Thinking":

#### Phase Implementation

1. **Question (Q)** - Core Question Extraction
   - Extracts THE single most important question from any input
   - Uses low temperature (0.3) for focused extraction
   - Handles questions, statements, topics, requests uniformly

2. **Abduction (A)** - Hypothesis Generation  
   - Generates 3 specific hypotheses to answer the core question
   - Default temperature 0.8 (user adjustable with --temperature)
   - Each hypothesis directly addresses the question

3. **Deduction (D)** - Evaluation & Answer
   - Evaluates each hypothesis on 5 criteria:
     - **Novelty** (20%): Innovation/uniqueness
     - **Impact** (30%): Positive change potential
     - **Cost** (20%): Resource efficiency (inverted)
     - **Feasibility** (20%): Implementation practicality
     - **Risks** (10%): Risk level (inverted)
   - Provides THE definitive answer with action plan
   - Uses low temperature (0.2) for analytical precision

4. **Induction (I)** - Verification
   - Verifies the answer with 3 real-world examples
   - Confirms broad applicability
   - Medium temperature (0.5) for balanced examples

#### Key Improvements

- **Universal Prompts**: Single set works for all input types
- **Phase Optimization**: Each phase uses ideal hyperparameters
- **Unified Scoring**: Evolution uses same criteria as deduction
- **User Control**: Temperature adjustment for creativity tuning

### Key Patterns

#### Agent Registration
```python
# Agents register themselves on import
@agent_registry.register_agent(ThinkingMethod.QUESTIONING)
class MyQuestioningAgent(ThinkingAgentInterface):
    # Implementation
```

#### LLM Integration
```python
# Unified interface for all LLM providers
provider = get_llm_provider("google")  # or "openai", "anthropic"
response = await provider.generate(request)
```

#### Error Handling
```python
# Circuit breaker pattern for fault tolerance
circuit_breaker = self._get_circuit_breaker(method)
if not circuit_breaker.can_call():
    return fallback_result
```

#### Simple QADI Usage
```python
# New simplified orchestrator
from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator

# Create orchestrator with optional temperature override
orchestrator = SimpleQADIOrchestrator(temperature_override=1.2)

# Run complete QADI cycle
result = await orchestrator.run_qadi_cycle(
    user_input="How to reduce employee turnover?",
    context="Tech startup with 50 employees"
)

# Access results
print(f"Core Question: {result.core_question}")
print(f"Best Answer: {result.final_answer}")
print(f"Action Plan: {result.action_plan}")
print(f"Verification: {result.verification_conclusion}")
```

#### Unified Evaluation
```python
from mad_spark_alt.core.unified_evaluator import UnifiedEvaluator

evaluator = UnifiedEvaluator()

# Evaluate a hypothesis
evaluation = await evaluator.evaluate_hypothesis(
    hypothesis="Implement flexible work arrangements",
    context="Reduce employee turnover",
    core_question="What causes employees to leave?"
)

# Access scores
print(f"Novelty: {evaluation.scores['novelty']}")
print(f"Impact: {evaluation.scores['impact']}")
print(f"Overall: {evaluation.overall_score}")
```

## Code Standards

### Python Code Style

- **PEP 8** compliance (enforced by `black`)
- **Type hints** for all functions and methods
- **Docstrings** for all public APIs (Google style)
- **Async/await** for all I/O operations

### Type Hints Guidelines

```python
# Good: Specific types with TypedDict
class MetricsDict(TypedDict):
    score: float
    confidence: float

def analyze_text(text: str) -> MetricsDict:
    return MetricsDict(score=0.8, confidence=0.9)

# Avoid: Generic Any types
def analyze_text(text: str) -> Dict[str, Any]:  # ❌ Too generic
```

### Constructor Annotations
```python
# All constructors must have explicit return type
def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
    self.config = config or {}
```

### Naming Conventions

- **Classes**: PascalCase (`SmartQADIOrchestrator`)
- **Functions/Methods**: snake_case (`run_qadi_cycle`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_RETRIES`)
- **Private methods**: Leading underscore (`_internal_method`)

### File Organization

```
src/mad_spark_alt/
├── core/           # Core system components
├── agents/         # QADI thinking agents
├── layers/         # Evaluation system
├── evolution/      # Genetic algorithm components
└── utils/          # Shared utilities
```

## Testing Guidelines

### Test Structure

- **Unit tests**: Test individual functions/methods
- **Integration tests**: Test component interactions
- **System tests**: Test complete workflows
- **Performance tests**: Measure execution time and memory

### Test Patterns

#### Testing Async Code
```python
@pytest.mark.asyncio
async def test_async_operation():
    result = await some_async_function()
    assert result.success
```

#### Mocking LLM Calls
```python
@pytest.fixture
def mock_llm_provider():
    with patch('module.get_llm_provider') as mock:
        mock.return_value.generate.return_value = mock_response
        yield mock
```

#### Testing with Variance
```python
# For genetic algorithms, test with tolerance
def test_evolution_improves_fitness():
    initial_fitness = calculate_fitness(population)
    evolved_population = evolve(population, generations=5)
    final_fitness = calculate_fitness(evolved_population)
    
    # Allow for randomness in genetic algorithms
    assert final_fitness >= initial_fitness * 0.9
```

### Test Coverage

- **Minimum**: 80% overall coverage
- **Critical paths**: 95%+ coverage
- **New features**: 100% coverage required

### Running Tests

```bash
# All tests
uv run pytest

# Specific test file
uv run pytest tests/test_specific.py -v

# With coverage report
uv run pytest --cov=src/mad_spark_alt --cov-report=html

# Performance tests
uv run pytest tests/performance/ --benchmark-only
```

## Performance Considerations

### Async Best Practices

1. **Use asyncio.gather()** for parallel operations
2. **Implement semaphores** for concurrency control
3. **Use asyncio.wait()** for timeout handling with partial results
4. **Shield critical tasks** to prevent cancellation

```python
# Good: Parallel with concurrency limit
semaphore = asyncio.Semaphore(max_concurrent)
tasks = [bounded_process(item) for item in items]
results = await asyncio.gather(*tasks)

# Good: Timeout with partial results
done, pending = await asyncio.wait(
    tasks, timeout=timeout, return_when=asyncio.ALL_COMPLETED
)
```

### Memory Management

- **Use generators** for large datasets
- **Implement caching** for expensive operations
- **Clean up resources** in finally blocks
- **Monitor memory usage** in long-running processes

### LLM API Optimization

- **Batch requests** when possible
- **Implement rate limiting** to avoid API limits
- **Use retry with exponential backoff**
- **Cache responses** for identical requests

## Contribution Workflow

### Git Workflow

1. **Create feature branch** from main
2. **Make focused commits** with clear messages
3. **Run all tests** before pushing
4. **Create pull request** with description
5. **Address review feedback**
6. **Squash merge** when ready

### Commit Message Format

```
type: brief description

- Detailed explanation of changes
- Why the change was needed
- Any breaking changes or migration notes

Resolves: #issue-number
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `style`, `perf`

### Pull Request Guidelines

1. **Clear title** describing the change
2. **Detailed description** with context
3. **Test coverage** for new functionality
4. **Documentation updates** if needed
5. **Performance impact** assessment

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Type checking passes (mypy)
- [ ] Documentation is updated
- [ ] Performance impact considered
- [ ] Security implications reviewed

## Debugging and Troubleshooting

### Common Issues

#### 1. LLM API Failures
```python
# Enable debug logging
import logging
logging.getLogger('mad_spark_alt').setLevel(logging.DEBUG)

# Check rate limiting
provider = get_llm_provider("google")
status = await provider.get_status()
```

#### 2. Circuit Breaker Issues
```python
# Check circuit breaker state
orchestrator = SmartQADIOrchestrator()
status = orchestrator.get_circuit_breaker_status()
```

#### 3. Test Failures
```python
# Use pytest debugging
pytest tests/test_file.py::test_function -v -s --pdb

# Check for timing issues in async tests
@pytest.mark.asyncio
async def test_with_proper_awaits():
    await asyncio.sleep(0.1)  # Allow async operations to complete
```

### Performance Profiling

```bash
# Profile memory usage
python -m memory_profiler script.py

# Profile execution time
python -m cProfile -o profile.stats script.py

# Analyze async performance
python -c "
import asyncio
import cProfile
asyncio.run(main(), debug=True)
"
```

### Logging Configuration

```python
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Component-specific logging
logging.getLogger('mad_spark_alt.core.llm_provider').setLevel(logging.DEBUG)
```

## Development Tools

### IDE Setup

#### VS Code
- Install Python extension
- Configure settings for black, mypy, isort
- Use pytest integration

#### PyCharm
- Configure interpreter to use uv environment
- Enable type checking
- Setup test runner for pytest

### Git Hooks

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Manual run
pre-commit run --all-files
```

## Security Considerations

1. **Never commit API keys** to repository
2. **Use environment variables** for sensitive data
3. **Validate all inputs** from external sources
4. **Implement rate limiting** for public APIs
5. **Regular dependency updates** for security patches

## Documentation Standards

- **README.md**: Quick start and overview
- **DEVELOPMENT.md**: This file - comprehensive dev guide
- **API documentation**: Generated from docstrings
- **Architecture docs**: High-level design decisions
- **Session handovers**: Track development progress

### Docstring Format

```python
def complex_function(param1: str, param2: Optional[int] = None) -> Dict[str, Any]:
    """
    Brief description of what the function does.

    Args:
        param1: Description of parameter
        param2: Optional parameter with default value

    Returns:
        Dictionary containing the results

    Raises:
        ValueError: When input validation fails
        
    Example:
        >>> result = complex_function("test", 42)
        >>> assert result["success"] is True
    """
```

---

For questions or clarifications, see the [README.md](README.md) or create an issue in the repository.