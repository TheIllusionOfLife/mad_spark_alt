# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mad Spark Alt is a sophisticated Multi-Agent Idea Generation System powered by Large Language Models and based on the QADI (Question → Abduction → Deduction → Induction) methodology from "Shin Logical Thinking".

### Current State: Advanced LLM-Powered System ✅
The system provides an intelligent multi-agent framework with:
1. **Smart QADI Orchestration** - AI-powered workflows with LLM integration
2. **Dynamic Prompt Engineering** - Auto-detects question types with 100% accuracy
3. **LLM-Only Agent System** - Template agents are meaningless and should NEVER be used
4. **Multi-Provider Support** - OpenAI, Anthropic, and Google API integration
5. **Cost-Aware Processing** - Real-time LLM usage tracking and optimization
6. **Google API Priority** - Always use real LLM APIs for meaningful insights

## ⚠️ CRITICAL: Never Use Template Agents

**Template agents produce generic, meaningless responses that don't engage with the actual question.**

- ❌ NEVER use `qadi_working.py` or any template-only implementations
- ✅ ALWAYS use tools that leverage Google API or other LLMs
- ✅ Use `qadi_simple_multi.py` for multi-agent analysis with Google API
- ✅ Use `qadi.py` for quick single-prompt analysis with Google API

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

# Run QADI with Google API (RECOMMENDED)
uv run python qadi_simple_multi.py "Your question here"  # Multi-agent with auto-detection
uv run python qadi_simple_multi.py --type=business "Your question"  # Force business perspective
uv run python qadi_simple_multi.py --concrete "Build something"  # Implementation-focused
uv run python qadi.py "Your question here"  # Simple version

# ❌ NEVER run template demos - they produce meaningless output
# DO NOT USE: qadi_working.py or template-only demos
```

### API Key Required for LLM Features
```bash
export GOOGLE_API_KEY="your-key"
```

## Architecture Overview

### Core Components

1. **LLM-Only Agent System**
   - Each thinking method MUST use LLM implementations
   - Template agents are meaningless - NEVER use them
   - LLM agents: Sophisticated reasoning, require API keys
   - Always prioritize Google API or other LLM providers

2. **Smart Orchestration** (`core/smart_orchestrator.py`)
   - `SmartQADIOrchestrator`: Manages intelligent agent selection
   - `SmartAgentRegistry`: Handles registration and preferences
   - Cost tracking across all LLM calls
   - Enhanced context building between phases

3. **LLM Integration** (`core/llm_provider.py`)
   - Google Gemini API integration
   - Robust JSON parsing with markdown extraction
   - Cost calculation and tracking
   - Retry logic with exponential backoff

4. **Registry Pattern** (`core/registry.py`, `core/smart_registry.py`)
   - Dynamic agent registration at runtime
   - Method-based agent retrieval
   - Plugin architecture for extensibility

5. **Dynamic Prompt Engineering** (`core/prompt_classifier.py`, `core/adaptive_prompts.py`)
   - `PromptClassifier`: Intelligent question type detection with word boundary matching
   - `AdaptivePromptGenerator`: Domain-specific prompts for each question type
   - Auto-detects: Technical, Business, Creative, Research, Planning, Personal
   - 100% accuracy on common question types with manual override support

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
   - **Centralized in cost_utils.py**: All cost calculations use unified functions
   - **Use ModelConfig directly**: `calculate_llm_cost_from_config()` avoids model name mismatch issues  
   - Track at individual API call level and distribute costs across generated ideas
   - Report total costs in orchestration results

4. **Error Handling**
   - Individual phase failures don't crash the cycle
   - Use simplified LLM calls when complex orchestration fails
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
│   ├── json_utils.py       # Robust JSON parsing
│   ├── prompt_classifier.py # Question type detection
│   └── adaptive_prompts.py # Domain-specific prompts
├── agents/                  # QADI thinking agents
│   ├── questioning/        # LLM agents only (ignore templates)
│   ├── abduction/          # Hypothesis generation
│   ├── deduction/          # Logical reasoning
│   └── induction/          # Pattern synthesis
└── layers/                  # Evaluation infrastructure
    ├── quantitative/       # Metrics-based evaluation
    └── llm_judges/         # AI-powered evaluation
```

### Testing Strategy
- Mock LLM API calls to avoid costs
- Test LLM agent paths only
- Verify API integration
- Check cost tracking accuracy

### CI/CD Pipeline
- GitHub Actions workflow in `.github/workflows/ci.yml`
- Tests across Python 3.8-3.11
- Uses `uv` for fast dependency management
- Includes CLI functionality tests
- Type checking and formatting validation

## Important Notes

1. **Always use `uv run`** prefix for CLI commands due to package installation method
2. **LLM agents require Google API key** - system MUST have GOOGLE_API_KEY to function properly
3. **Cost tracking is automatic** - check `result.llm_cost` after orchestration
4. **All operations are async** - use `asyncio.run()` for synchronous contexts
5. **JSON parsing is critical** - LLMs often return markdown-wrapped JSON

## Quick Verification
```bash
# Verify installation
uv run mad-spark --help

# Test QADI system with Google API
uv run python qadi_simple_multi.py "Test question"

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
- **Evaluation Metrics**: Count only new evaluations, excluding preserved elite individuals
- **Checkpoint Resume**: Be clear about state vs transition numbers to avoid off-by-one errors

### Type Safety Requirements
- **Critical Pattern**: Always handle Optional fields defensively
- **Convention**: Use `field if field is not None else default_value` pattern for clarity and safety
- **Example**: `(idea.confidence_score if idea.confidence_score is not None else 0.5)` in operators.py

### CI/CD Specifics
- **Common Failure**: mypy type checking errors (even when local passes)
- **Prevention**: Run `uv run mypy src/` before every push
- **Fix Pattern**: Check datetime fields, Optional annotations, type imports

### LLM Integration Patterns
- **LLM-Only System**: Only use LLM agents, never templates
- **Cost Tracking**: Distributed across generated ideas automatically
- **API Priority**: Google API is preferred for reliability

### Registry Architecture
- **Global Registry**: Use `agent_registry` and `evaluator_registry`
- **Dynamic Loading**: Agents register at module import time
- **Clear Before Tests**: Always `agent_registry.clear()` in test setup
- **CLI Environment Loading**: CLI must call `load_env_file()` before registry use

### QADI Orchestration
- **Phase Order**: Question → Abduction → Deduction → Induction
- **Parallel Execution**: Use `run_parallel_generation` for efficiency
- **Context Passing**: Each phase builds on previous phase results

### Import Requirements
- **No Inline Imports**: All imports must be at module level for CI
- **Relative Imports**: Use `from ...core import X` pattern
- **Type Imports**: Use `from typing import TYPE_CHECKING` for circular deps

### Deprecation Best Practices
- **Module-Level Warnings**: Issue deprecation warnings at module import time
- **Clear Migration Path**: Always specify what to use instead
- **Stack Level**: Use `stacklevel=2` for warnings to show caller location
- **Example**:
  ```python
  import warnings
  warnings.warn(
      "module_name is deprecated and will be removed in v2.0.0. "
      "Use NewModule instead.",
      DeprecationWarning,
      stacklevel=2
  )
  ```

### Performance Testing
- **Test Naming**: Avoid `test_*.py` pattern in root tests/ (caught by .gitignore)
- **Performance Files**: Use descriptive names like `performance_benchmarks.py`
- **Memory Tracking**: Use `tracemalloc` for memory usage analysis
- **Fixture Pattern**: Use `@pytest.fixture(autouse=True)` for setup/teardown

### Claude Code Custom Commands
- **$ARGUMENTS Support**: Custom commands can accept arguments via `$ARGUMENTS`
- **Example**: `/fix_pr_since_commit 1916eed` passes "1916eed" as $ARGUMENTS
- **Implementation**: Use `${ARGUMENTS:-default_value}` for optional parameters
- **Usage**: Enables flexible command reuse without duplication

### GeneticAlgorithm API
- **Constructor**: Takes no arguments - `GeneticAlgorithm()`
- **Evolution**: Use `EvolutionRequest` object with `evolve()` method
- **Result Access**: Use `result.final_population` not direct population return