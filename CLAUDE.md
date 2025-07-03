# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TRANSFORMATION COMPLETE:** Mad Spark Alt has evolved from an AI creativity evaluation system into a **Multi-Agent Idea Generation System** based on "Shin Logical Thinking" QADI methodology.

### Current State: QADI System Implementation ✅
The system now provides a complete multi-agent framework with:
1. **QADI Cycle Orchestration** - Question → Abduction → Deduction → Induction workflows ✅
2. **Thinking Method Agents** - Specialized AI agents for different cognitive approaches ✅  
3. **Unified Registry System** - Seamless management of evaluators and thinking agents ✅
4. **Async Processing Framework** - Efficient multi-agent coordination ✅
5. **Creativity Evaluation Engine** - Multi-dimensional assessment for fitness scoring ✅

### Future Evolution: Genetic Algorithm Integration 🚧
The next phase will add:
1. **Genetic Evolution Engine** - Idea population evolution through genetic algorithms
2. **Human-AI Collaboration Interface** - Interactive ideation sessions and feedback loops
3. **Advanced Fitness Functions** - Sophisticated evaluation criteria for idea evolution

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

# List available evaluators and agents
mad-spark list-evaluators

# List LLM judges
mad-spark list-judges

# Test LLM connections
mad-spark test-judges

# Evaluate single text
mad-spark evaluate "creative text" --model gpt-4

# LLM judge evaluation
mad-spark evaluate "text" --llm-judge gpt-4

# Multi-judge jury evaluation
mad-spark evaluate "text" --jury "gpt-4,claude-3-sonnet,gemini-pro"

# Pre-configured jury budgets
mad-spark evaluate "text" --jury-budget balanced
```

### QADI System Usage
```bash
# Run QADI demonstration
python examples/qadi_demo.py

# Run basic examples
python examples/basic_usage.py

# Test the complete system
uv run pytest tests/test_qadi_system.py -v

# Test individual agents
python -c "
import asyncio
from mad_spark_alt.agents import QuestioningAgent
from mad_spark_alt.core import IdeaGenerationRequest

async def test():
    agent = QuestioningAgent()
    request = IdeaGenerationRequest(
        problem_statement='How can we improve urban sustainability?',
        context='Focus on practical solutions',
        max_ideas_per_method=3
    )
    result = await agent.generate_ideas(request)
    for idea in result.generated_ideas:
        print(f'💡 {idea.content}')

asyncio.run(test())
"
```

### Environment Variables
LLM Judge functionality requires API keys:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key" 
export GOOGLE_API_KEY="your-google-key"
```

Current implementation primarily uses local evaluation methods.

## System Architecture (Current Implementation)

### ✅ Phase 1: Core Architecture (COMPLETED)
**Multi-Agent QADI Framework**

1. **Core Interfaces** (`core/interfaces.py`)
   - ✅ `ThinkingAgentInterface` - Common interface for all thinking agents
   - ✅ `IdeaGenerationRequest` - Standardized input for idea generation
   - ✅ `GeneratedIdea` - Rich idea representation with metadata
   - ✅ `ThinkingMethod` enum - QUESTIONING, ABDUCTION, DEDUCTION, INDUCTION
   - ✅ `IdeaGenerationResult` - Structured output from agents

2. **Unified Registry System** (`core/registry.py`)
   - ✅ `ThinkingAgentRegistry` - Agent management and discovery
   - ✅ `UnifiedRegistry` - Seamless evaluator and agent integration
   - ✅ Dynamic registration with convenience functions
   - ✅ Method-based agent retrieval and indexing

3. **QADI Orchestration Engine** (`core/orchestrator.py`)
   - ✅ `QADIOrchestrator` - Multi-phase cycle coordination
   - ✅ Sequential and parallel agent processing
   - ✅ Enhanced context building between phases
   - ✅ Robust error handling for missing agents
   - ✅ Idea synthesis and aggregation

### ✅ Phase 2: Thinking Method Agents (COMPLETED)
**"Shin Logical Thinking" Implementation**

1. **Questioning Agent** (`agents/questioning/`)
   - ✅ Diverse questioning strategies (clarifying, alternative, challenging, etc.)
   - ✅ Problem framing and assumption questioning
   - ✅ Context-aware question generation

2. **Abductive Agent** (`agents/abduction/`)
   - ✅ Hypothesis generation through creative leaps
   - ✅ Causal, analogical, and pattern-based reasoning
   - ✅ "What if" scenario exploration

3. **Deductive Agent** (`agents/deduction/`)
   - ✅ Logical validation and systematic reasoning
   - ✅ Structured consequence derivation
   - ✅ Constraint-based analysis

4. **Inductive Agent** (`agents/induction/`)
   - ✅ Pattern synthesis and rule formation
   - ✅ Generalization from specific observations
   - ✅ Meta-pattern recognition and insight extraction

### 🚧 Phase 3: Genetic Evolution Engine (PLANNED)
**Idea Population Evolution**

1. **Evolution Engine** (`evolution/genetic_algorithm.py`)
   - Genetic operators for idea crossover and mutation
   - Population management and selection strategies
   - Fitness evaluation using existing creativity metrics

2. **Human-AI Collaboration** (`collaboration/interface.py`)
   - Interactive ideation sessions
   - Real-time feedback integration
   - Collaborative idea refinement

## Implementation Architecture

### ✅ Current Structure (Fully Implemented QADI System)
```
src/mad_spark_alt/
├── core/                        # ✅ Core system components
│   ├── orchestrator.py         # ✅ QADI cycle coordination engine
│   ├── interfaces.py           # ✅ Agent and evaluator interfaces
│   ├── registry.py             # ✅ Unified agent/evaluator management
│   └── evaluator.py            # ✅ Creativity evaluation engine
├── agents/                      # ✅ QADI thinking method agents
│   ├── questioning/            # ✅ Question generation and framing
│   │   └── agent.py           # ✅ QuestioningAgent implementation
│   ├── abduction/              # ✅ Hypothesis generation and creative leaps
│   │   └── agent.py           # ✅ AbductionAgent implementation
│   ├── deduction/              # ✅ Logical validation and reasoning
│   │   └── agent.py           # ✅ DeductionAgent implementation
│   └── induction/              # ✅ Pattern synthesis and generalization
│       └── agent.py           # ✅ InductionAgent implementation
├── layers/                      # ✅ Evaluation infrastructure
│   ├── quantitative/           # ✅ Automated metrics (diversity, quality)
│   ├── llm_judges/             # ✅ AI-powered evaluation
│   └── human_eval/             # ✅ Human assessment interface
├── examples/                    # ✅ Usage demonstrations
│   ├── qadi_demo.py            # ✅ Complete QADI system demo
│   └── basic_usage.py          # ✅ Basic evaluation examples
├── tests/                       # ✅ Comprehensive test suite
│   ├── test_qadi_system.py     # ✅ QADI agents and orchestration tests
│   └── unit/                   # ✅ Unit tests for components
└── cli.py                       # ✅ Command-line interface
```

### 🚧 Future Enhancements (Genetic Evolution)
```
src/mad_spark_alt/
├── evolution/                   # 🚧 Genetic algorithm implementation
│   ├── genetic_algorithm.py    # 🚧 Population evolution engine
│   ├── fitness.py              # 🚧 Idea fitness evaluation
│   └── operators.py            # 🚧 Crossover, mutation, selection
├── collaboration/               # 🚧 Human-AI interaction
│   ├── interface.py            # 🚧 Interactive ideation sessions
│   └── feedback.py             # 🚧 Human feedback integration
└── web/                         # 🚧 Web interface (optional)
    ├── api.py                  # 🚧 REST API for remote access
    └── dashboard.py            # 🚧 Real-time monitoring dashboard
```

### Key Classes

**Core Interfaces** (`core/interfaces.py`):
- `ThinkingAgentInterface`: Abstract base for all thinking agents ✅
- `EvaluatorInterface`: Abstract base for all evaluators ✅
- `IdeaGenerationRequest`: Input data structure for idea generation ✅
- `IdeaGenerationResult`: Output data structure from agents ✅
- `GeneratedIdea`: Rich idea representation with metadata ✅
- `ThinkingMethod`: Enum (QUESTIONING, ABDUCTION, DEDUCTION, INDUCTION) ✅
- `EvaluationRequest`: Input data structure for evaluation ✅
- `EvaluationResult`: Output data structure from evaluators ✅
- `ModelOutput`: Represents AI-generated content to evaluate ✅

**QADI Orchestration** (`core/orchestrator.py`):
- `QADIOrchestrator`: Coordinates multi-phase thinking cycles ✅
- `QADICycleResult`: Complete cycle result with phase breakdowns ✅
- Handles sequential and parallel agent processing ✅
- Enhanced context building between phases ✅

**Registry System** (`core/registry.py`):
- `ThinkingAgentRegistry`: Dynamic agent registration and management ✅
- `EvaluatorRegistry`: Dynamic evaluator registration ✅ 
- `UnifiedRegistry`: Seamless integration of both systems ✅
- Method-based agent retrieval and discovery ✅

**Creativity Evaluation** (`core/evaluator.py`):
- `CreativityEvaluator`: Coordinates evaluation across all layers ✅
- Handles async execution, result aggregation, scoring ✅

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

# Test basic evaluation
mad-spark evaluate "The AI dreamed of electric sheep in quantum meadows."

# Test core functionality without API keys
uv run pytest tests/unit/test_core.py -v

# Run all tests
uv run pytest

# Check registry state (useful for debugging)
python -c "from mad_spark_alt.core import registry; print([e.name for e in registry.get_evaluators()])"
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