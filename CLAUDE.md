# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**INTELLIGENT SYSTEM COMPLETE:** Mad Spark Alt is a **sophisticated Multi-Agent Idea Generation System** powered by Large Language Models and based on "Shin Logical Thinking" QADI methodology.

### Current State: Advanced LLM-Powered System ✅
The system provides an intelligent multi-agent framework with:
1. **Smart QADI Orchestration** - AI-powered Question → Abduction → Deduction → Induction workflows ✅
2. **LLM-Powered Agents** - Sophisticated AI agents using OpenAI, Anthropic, and Google APIs ✅  
3. **Intelligent Agent Registry** - Automatic LLM preference with graceful template fallback ✅
4. **Cost-Aware Processing** - Real-time LLM cost tracking and optimization ✅
5. **Multi-Provider Support** - Seamless integration across multiple LLM providers ✅
6. **Robust Fallback System** - Automatic degradation to template agents when needed ✅
7. **Creativity Evaluation Engine** - Multi-dimensional assessment for idea fitness scoring ✅

### Next Evolution: Advanced AI Integration 🚧
Future enhancements will include:
1. **Genetic Algorithm Engine** - Evolution of LLM-generated ideas through fitness-based selection
2. **Context-Aware Processing** - Domain knowledge integration and specialized reasoning
3. **Human-AI Collaboration** - Interactive ideation with intelligent agent assistance
4. **Advanced Analytics** - Multi-dimensional creativity and feasibility assessment

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
**IMPORTANT**: CLI commands must be run with `uv run` prefix due to package installation method.

```bash
# Install package first (required for CLI access)
uv pip install -e .

# Main CLI entry point
uv run mad-spark --help

# List available evaluators
uv run mad-spark list-evaluators

# Evaluate single text for creativity
uv run mad-spark evaluate "The AI dreamed of electric sheep in quantum meadows."

# Batch evaluate multiple files
uv run mad-spark batch-evaluate file1.txt file2.txt file3.txt

# Compare creativity of multiple responses
uv run mad-spark compare "idea1" "idea2" "idea3"
```

**Note**: The CLI currently supports evaluation functionality. QADI generation is available through Python API only.

### Smart QADI System Usage (Python API)
**IMPORTANT**: QADI generation now uses intelligent LLM-powered agents with automatic fallback.

```bash
# Set API keys for LLM-powered generation (recommended)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key" 
export GOOGLE_API_KEY="your-google-key"

# Run Smart QADI demonstration (shows LLM vs template agents)
uv run python examples/qadi_demo.py

# Run LLM showcase demo (requires API keys)
uv run python examples/llm_showcase_demo.py

# Run LLM-specific agent demos
uv run python examples/llm_questioning_demo.py
uv run python examples/llm_abductive_demo.py

# Test the intelligent system
uv run pytest tests/test_qadi_system.py -v

# Test smart agent setup
uv run python -c "
import asyncio
from mad_spark_alt.core import SmartQADIOrchestrator

async def test():
    orchestrator = SmartQADIOrchestrator()
    result = await orchestrator.run_qadi_cycle(
        problem_statement='How can we improve urban sustainability?',
        context='Focus on practical, innovative solutions',
        cycle_config={'max_ideas_per_method': 3}
    )
    print(f'Generated {len(result.synthesized_ideas)} ideas')
    print(f'Agent types: {set(result.agent_types.values())}')
    if result.llm_cost > 0:
        print(f'LLM cost: \${result.llm_cost:.4f}')
    for phase, phase_result in result.phases.items():
        print(f'{phase.upper()}: {len(phase_result.generated_ideas)} ideas')

asyncio.run(test())
"
```

**Expected Output**: The Smart QADI system produces sophisticated AI-powered ideas:
- **With API Keys**: Context-aware, intelligent reasoning with detailed explanations
- **Without API Keys**: Graceful fallback to template-based generation
- **Cost Tracking**: Real-time monitoring of LLM usage and costs
- **Agent Types**: Clear indication of LLM vs template agent usage

This is intelligent Phase 2+ behavior - AI-powered reasoning with robust fallbacks.

### Environment Variables
LLM Judge functionality requires API keys:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key" 
export GOOGLE_API_KEY="your-google-key"
```

Current implementation primarily uses local evaluation methods.

## Current System State & Expectations

### ✅ What Works Now (Phase 2+ - LLM-Powered)
- **Smart QADI Orchestration**: All 4 phases use intelligent LLM reasoning with fallback
- **LLM-Powered Generation**: Agents use sophisticated AI reasoning from OpenAI, Anthropic, Google
- **Intelligent Agent Selection**: Automatic preference for LLM agents with graceful template fallback
- **Cost-Aware Processing**: Real-time LLM usage tracking and cost optimization
- **System Integration**: Smart registry, orchestrator, and agents work seamlessly
- **Evaluation Engine**: Multi-dimensional creativity assessment with quantitative metrics
- **CLI Interface**: Basic evaluation commands (generation via Python API)

### 🔍 Expected Output Characteristics
The Smart QADI system produces **intelligent AI-powered results**:

**With LLM Agents**:
- **Questioning**: "What stakeholder perspectives haven't been considered in urban sustainability? How might economic incentives conflict with environmental goals?"
- **Abduction**: "This problem might stem from systemic misalignment between short-term economic pressures and long-term sustainability benefits, suggesting we need innovative financing mechanisms..."
- **Deduction**: "If we implement carbon pricing mechanisms, then logically we must also provide transition support for affected industries to prevent economic disruption while achieving environmental goals."
- **Induction**: "Analyzing patterns across successful sustainability initiatives reveals that effective solutions typically integrate economic incentives, stakeholder engagement, and measurable impact metrics."

**This represents sophisticated AI reasoning** - context awareness, stakeholder analysis, and domain-specific insights.

### 🚧 What's Coming Next (Future Phases)
- **Genetic Algorithm Engine**: Evolution of LLM-generated ideas through fitness-based selection
- **Context-Aware Processing**: Domain knowledge integration and specialized reasoning strategies
- **Human-AI Collaboration**: Interactive ideation sessions with intelligent agent assistance
- **Advanced Analytics**: Multi-dimensional creativity and feasibility assessment frameworks

### 📋 Key Files for New Sessions
- **USER_GUIDE.md**: Complete user experience guide with LLM and template agent examples
- **examples/qadi_demo.py**: Smart QADI demonstration with LLM vs template comparison
- **examples/llm_showcase_demo.py**: Advanced LLM agent capabilities showcase
- **src/mad_spark_alt/core/smart_registry.py**: Intelligent agent registration system
- **src/mad_spark_alt/core/smart_orchestrator.py**: Smart QADI orchestration with LLM preference
- **src/mad_spark_alt/agents/*/llm_agent.py**: LLM-powered agent implementations
- **tests/test_qadi_system.py**: Verification that all components work correctly

## System Architecture (Current Implementation)

### ✅ Phase 2+: Advanced LLM-Powered Architecture (COMPLETED)
**Intelligent Multi-Agent QADI Framework**

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
# Install package first
uv pip install -e .

# Verify installation and basic functionality
uv run mad-spark --help
uv run mad-spark list-evaluators

# Test basic evaluation
uv run mad-spark evaluate "The AI dreamed of electric sheep in quantum meadows."

# Test QADI system (template-based generation)
uv run python examples/qadi_demo.py

# Test core functionality without API keys
uv run pytest tests/unit/test_core.py -v

# Run all tests
uv run pytest

# Check registry state (useful for debugging)
uv run python -c "from mad_spark_alt.core import registry; print([e.name for e in registry.get_evaluators()])"
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