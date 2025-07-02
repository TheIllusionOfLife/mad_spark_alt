# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TRANSFORMATION IN PROGRESS:** Mad Spark Alt is evolving from an AI creativity evaluation system into a **Multi-Agent Idea Generation System** based on "Shin Logical Thinking" methodology.

### Current State: Evaluation Infrastructure Foundation
The existing codebase provides a solid foundation with:
1. **Plugin Registry System** - Perfect for managing thinking method agents
2. **Async Processing Framework** - Essential for multi-agent coordination  
3. **Evaluation Infrastructure** - Will be repurposed for genetic algorithm fitness evaluation

### Target Architecture: Multi-Agent Idea Generation
The system will transform into a collaborative idea generation platform with:
1. **QADI Cycle Orchestration** - Question → Abduction → Deduction → Induction workflows
2. **Thinking Method Agents** - Specialized AI agents for different cognitive approaches
3. **Genetic Evolution Engine** - Idea population evolution through genetic algorithms
4. **Human-AI Collaboration** - Interactive ideation sessions and feedback loops

## Commands

### Development Environment
```bash
# Install dependencies (preferred)
uv sync

# Install with dev dependencies
uv sync --dev

# Alternative with pip
pip install -e .

# Run development server (if needed)
python main.py
```

### Testing & Quality
```bash
# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/unit/test_llm_judges.py

# Run tests with coverage
uv run pytest --cov=src/mad_spark_alt --cov-report=html

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

## Transformation Roadmap

### Phase 1: Core Architecture Evolution (Current Priority)
**Transform Evaluation → Generation Framework**

1. **Extend Core Interfaces** (`core/interfaces.py`)
   - Create `ThinkingAgentInterface` (extends `EvaluatorInterface` pattern)
   - Add `IdeaGenerationRequest` (based on `EvaluationRequest`)
   - Define `GeneratedIdea` (extends `ModelOutput` with idea metadata)
   - Implement `ThinkingMethod` enum (QADI, ABDUCTION, DEDUCTION, INDUCTION)

2. **Expand Registry System** (`core/registry.py`)
   - Add `ThinkingAgentRegistry` for cognitive agent management
   - Support thinking method indexing and orchestration
   - Dynamic agent discovery and collaboration patterns

3. **Build Orchestration Engine** (`core/orchestrator.py`)
   - Multi-agent coordination using existing async patterns
   - QADI cycle implementation (Question → Abduction → Deduction → Induction)
   - Agent communication and idea synthesis protocols

### Phase 2: Thinking Method Agents (Next Sprint)
**Implement "Shin Logical Thinking" Methods**

1. **Questioning Agent** (`agents/questioning/`)
   - Diverse questioning techniques and problem framing
   - Uses existing LLM integration patterns

2. **Abductive Agent** (`agents/abduction/`)
   - Hypothesis generation and creative leaps
   - Pattern recognition from observations

3. **Deductive Agent** (`agents/deduction/`)
   - Logical validation and systematic reasoning
   - Structured consequence derivation

4. **Inductive Agent** (`agents/induction/`)
   - Pattern generalization and rule formation
   - Creative synthesis and insight extraction

### Phase 3: Genetic Evolution Engine (Following Sprint)
**Leverage Current Evaluation for Idea Fitness**

1. **Evolution Engine** (`evolution/genetic_algorithm.py`)
   - Repurpose existing evaluation framework for fitness scoring
   - Implement crossover, mutation, and selection operators

2. **Human-AI Collaboration** (`collaboration/interface.py`)
   - Interactive ideation sessions
   - Real-time feedback integration

## Current Architecture (Foundation Layer)

### Current Structure (Foundation)
```
src/mad_spark_alt/
├── core/                    # ✅ Evaluation engine (foundation for orchestration)
│   ├── evaluator.py        # ✅ CreativityEvaluator (will become fitness evaluator)
│   ├── interfaces.py       # ✅ Abstract base classes (will extend for agents)
│   └── registry.py         # ✅ Plugin system (perfect for agent management)
├── layers/                 # ✅ Current evaluation layers (fitness evaluation)
│   ├── quantitative/       # ✅ Automated metrics (idea fitness scoring)
│   │   ├── diversity.py    # ✅ Diversity metrics (idea novelty scoring)
│   │   └── quality.py      # ✅ Quality metrics (idea feasibility scoring)
│   └── human_eval/         # ✅ Human assessment (collaboration interface)
├── models/                 # ✅ Data models (will extend for ideas)
├── storage/                # ✅ Persistence layer
└── cli.py                  # ✅ CLI interface (will extend for generation)
```

### Target Structure (Multi-Agent Generation System)
```
src/mad_spark_alt/
├── core/                    # Core orchestration and coordination
│   ├── orchestrator.py     # 🚧 Multi-agent coordination engine
│   ├── interfaces.py       # 🚧 Extended with ThinkingAgentInterface
│   ├── registry.py         # 🚧 Enhanced for agent management
│   └── evaluator.py        # ✅ Fitness evaluation (repurposed)
├── agents/                 # 🚧 Thinking method agent implementations
│   ├── questioning/        # 🚧 Question generation and framing
│   ├── abduction/          # 🚧 Hypothesis generation and creative leaps
│   ├── deduction/          # 🚧 Logical validation and reasoning
│   └── induction/          # 🚧 Pattern synthesis and generalization
├── evolution/              # 🚧 Genetic algorithm implementation
│   ├── genetic_algorithm.py # 🚧 Population evolution engine
│   ├── fitness.py          # 🚧 Idea fitness evaluation (uses existing metrics)
│   └── operators.py        # 🚧 Crossover, mutation, selection
├── collaboration/          # 🚧 Human-AI interaction
│   ├── interface.py        # 🚧 Interactive ideation sessions
│   └── feedback.py         # 🚧 Human feedback integration
├── layers/                 # ✅ Evaluation infrastructure (now fitness evaluation)
└── cli.py                  # 🚧 Extended CLI for generation workflows
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
from typing import Any, Dict, List
from mad_spark_alt.core import EvaluatorInterface, register_evaluator
from mad_spark_alt.core.interfaces import EvaluationLayer, EvaluationRequest, EvaluationResult, OutputType

class MyEvaluator(EvaluatorInterface):
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

### Quick Development Verification
```bash
# Verify installation and basic functionality
mad-spark --help
mad-spark list-evaluators

# Test core functionality without API keys
uv run pytest tests/unit/test_core.py -v
```

## Important Implementation Notes

### Cost Management
LLM judges track token usage and costs. Budget-aware jury configurations prevent runaway expenses.

### Multi-Judge Consensus
The jury system uses voting mechanisms to handle disagreements between AI judges, improving evaluation reliability.

### Extensibility
The plugin architecture allows easy addition of new evaluation methods without modifying core code.

### Performance
Async execution enables parallel evaluation across multiple AI services, significantly reducing total evaluation time.