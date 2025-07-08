# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mad Spark Alt is a sophisticated Multi-Agent Idea Generation System powered by Large Language Models and based on the QADI (Question → Abduction → Deduction → Induction) methodology from "Shin Logical Thinking".

### Current State: Advanced LLM-Powered System ✅
The system provides an intelligent multi-agent framework with:
1. **Smart QADI Orchestration** - AI-powered workflows with automatic LLM preference
2. **Dual Agent System** - LLM-powered agents with template-based fallbacks
3. **Multi-Provider Support** - OpenAI, Anthropic, and Google API integration
4. **Cost-Aware Processing** - Real-time LLM usage tracking and optimization
5. **Robust Error Handling** - Graceful degradation when APIs unavailable

## Commands

### Development Environment
```bash
# Install dependencies (preferred method)
uv sync

# Install with dev dependencies
uv sync --dev

# Install package in editable mode (required for CLI)
uv pip install -e .
```

### Testing & Quality
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_qadi_system.py -v

# Run with coverage
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
# Main CLI entry point
uv run mad-spark --help

# List available evaluators
uv run mad-spark list-evaluators

# Evaluate single text
uv run mad-spark evaluate "The AI dreamed of electric sheep"

# Run QADI demos
uv run python examples/qadi_demo.py
uv run python examples/llm_showcase_demo.py
```

### API Keys Required for LLM Features
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

## Architecture Overview

### Core Components

1. **Dual Agent System**
   - Each thinking method has template AND LLM implementations
   - Template agents: Fast, no dependencies, consistent output
   - LLM agents: Sophisticated reasoning, require API keys
   - Automatic preference for LLM with graceful fallback

2. **Smart Orchestration** (`core/smart_orchestrator.py`)
   - `SmartQADIOrchestrator`: Manages intelligent agent selection
   - `SmartAgentRegistry`: Handles registration and preferences
   - Cost tracking across all LLM calls
   - Enhanced context building between phases

3. **LLM Integration** (`core/llm_provider.py`)
   - Unified interface for OpenAI, Anthropic, Google
   - Robust JSON parsing with markdown extraction
   - Cost calculation and tracking
   - Retry logic with exponential backoff

4. **Registry Pattern** (`core/registry.py`, `core/smart_registry.py`)
   - Dynamic agent registration at runtime
   - Method-based agent retrieval
   - Plugin architecture for extensibility

### Key Implementation Patterns

1. **JSON Parsing Robustness**
   ```python
   # LLMs often return JSON wrapped in markdown
   def safe_json_parse(text: str, fallback: Optional[Dict] = None) -> Dict:
       # 1. Try direct JSON parsing
       # 2. Extract from markdown code blocks
       # 3. Use regex patterns
       # 4. Return fallback on all failures
   ```

2. **Agent Interface**
   ```python
   class ThinkingAgentInterface:
       @property
       def is_llm_powered(self) -> bool:
           """Distinguish LLM from template agents"""
       
       async def generate_ideas(self, request: IdeaGenerationRequest) -> IdeaGenerationResult:
           """All agents implement this async method"""
   ```

3. **Cost Tracking**
   - Track at individual API call level
   - Distribute costs across generated ideas
   - Report total costs in orchestration results

4. **Error Handling**
   - Individual phase failures don't crash the cycle
   - Automatic fallback to template agents
   - Comprehensive logging throughout

### Project Structure
```
src/mad_spark_alt/
├── core/                    # Core system components
│   ├── interfaces.py       # Common interfaces
│   ├── orchestrator.py     # Basic QADI orchestration
│   ├── smart_orchestrator.py # LLM-powered orchestration
│   ├── registry.py         # Agent/evaluator registry
│   ├── smart_registry.py   # Intelligent agent registry
│   ├── llm_provider.py     # LLM integration layer
│   └── json_utils.py       # Robust JSON parsing
├── agents/                  # QADI thinking agents
│   ├── questioning/        # Both template and LLM agents
│   ├── abduction/          # Hypothesis generation
│   ├── deduction/          # Logical reasoning
│   └── induction/          # Pattern synthesis
└── layers/                  # Evaluation infrastructure
    ├── quantitative/       # Metrics-based evaluation
    └── llm_judges/         # AI-powered evaluation
```

### Testing Strategy
- Mock LLM API calls to avoid costs
- Test both LLM and template agent paths
- Verify fallback behavior
- Check cost tracking accuracy

### CI/CD Pipeline
- GitHub Actions workflow in `.github/workflows/ci.yml`
- Tests across Python 3.8-3.11
- Uses `uv` for fast dependency management
- Includes CLI functionality tests
- Type checking and formatting validation

## Important Notes

1. **Always use `uv run`** prefix for CLI commands due to package installation method
2. **LLM agents require API keys** - system gracefully falls back to templates without them
3. **Cost tracking is automatic** - check `result.llm_cost` after orchestration
4. **All operations are async** - use `asyncio.run()` for synchronous contexts
5. **JSON parsing is critical** - LLMs often return markdown-wrapped JSON

## Quick Verification
```bash
# Verify installation
uv run mad-spark --help

# Test QADI system (shows LLM vs template comparison)
uv run python examples/qadi_demo.py

# Check available agents
uv run python -c "
from mad_spark_alt.core import SmartAgentRegistry
registry = SmartAgentRegistry()
print(f'Available methods: {list(registry._agents.keys())}')
"
```

## Project-Specific Patterns

### Evolution System Testing
- **Pattern**: Test genetic algorithms with variance tolerance, not exact values
- **Convention**: Use `assert final_fitness >= initial_fitness * 0.9` for randomness
- **Gotcha**: Perfect thinking method balance isn't guaranteed in evolution

### Type Safety Requirements
- **Critical Pattern**: Always handle Optional fields defensively
- **Convention**: Use `(field or default_value)` pattern throughout
- **Example**: `(idea.confidence_score or 0.5)` in operators.py

### CI/CD Specifics
- **Common Failure**: mypy type checking errors (even when local passes)
- **Prevention**: Run `uv run mypy src/` before every push
- **Fix Pattern**: Check datetime fields, Optional annotations, type imports

### LLM Integration Patterns
- **Dual Agent System**: Every thinking agent has LLM + template versions
- **Cost Tracking**: Distributed across generated ideas automatically
- **Fallback**: Graceful degradation when API keys missing

### Registry Architecture
- **Global Registry**: Use `agent_registry` and `evaluator_registry`
- **Dynamic Loading**: Agents register at module import time
- **Clear Before Tests**: Always `agent_registry.clear()` in test setup

### QADI Orchestration
- **Phase Order**: Question → Abduction → Deduction → Induction
- **Parallel Execution**: Use `run_parallel_generation` for efficiency
- **Context Passing**: Each phase builds on previous phase results

### Import Requirements
- **No Inline Imports**: All imports must be at module level for CI
- **Relative Imports**: Use `from ...core import X` pattern
- **Type Imports**: Use `from typing import TYPE_CHECKING` for circular deps