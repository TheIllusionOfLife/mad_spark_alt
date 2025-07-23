# Mad Spark Alt - AI-Powered QADI Analysis System

Intelligent analysis system using QADI methodology (Question → Abduction → Deduction → Induction) to provide structured, multi-perspective insights on any topic.

## Features

- **QADI Methodology**: Structured 4-phase analysis for any question or problem
- **Universal Evaluation**: Impact, Feasibility, Accessibility, Sustainability, Scalability
- **Multiple Analysis Modes**: Simple, hypothesis-driven, multi-perspective
- **Temperature Control**: Adjust creativity level (0.0-2.0)
- **Audience-Neutral**: Practical insights for everyone, not just businesses
- **Real-World Examples**: Concrete applications at individual, community, and systemic levels

## Installation

```bash
git clone https://github.com/TheIllusionOfLife/mad_spark_alt.git
cd mad_spark_alt
uv sync  # Or: pip install -e .
```

## Quick Start

```bash
# Setup API key (REQUIRED)
echo "GOOGLE_API_KEY=your_key_here" > .env
```

### Default Command (Recommended)

```bash
# Simple, clear analysis with improved Phase 1
uv run mad_spark_alt "How can we reduce plastic waste?"

# Add genetic evolution to optimize ideas
uv run mad_spark_alt "How can we reduce plastic waste?" --evolve

# Customize evolution parameters
uv run mad_spark_alt "Your question" --evolve --generations 5 --population 20
```

### Other Analysis Modes

```bash
# Simple QADI analysis with direct prompts
uv run python qadi_simple.py "Your question here"

# Multi-agent analysis with type detection
uv run python qadi_simple_multi.py "Your question here"

# Multi-perspective analysis (environmental/personal/technical/business/etc.)
uv run python qadi_multi_perspective.py "Your question here"
```

### Advanced Options

```bash
# Temperature control (creativity level)
uv run python qadi_simple.py "Your question" --temperature 1.2

# Evolution with custom parameters
uv run python qadi_simple.py "Your question" --evolve --generations 10 --population 15

# Multi-perspective with forced perspectives
uv run python qadi_multi_perspective.py "Your question" --perspectives environmental,technical

# Note: Evolution arguments require --evolve flag
# This will show helpful error: uv run python qadi_simple.py "question" --generations 5
```

### Example Prompts

**Business**: "How can small businesses compete with large corporations?"
**Technology**: "How can AI improve rural healthcare accessibility?"
**Environment**: "How can cities become carbon-neutral by 2030?"
**Creative**: "What if gravity worked differently on weekends?"

### CLI Tools

```bash
# Evaluate creativity
uv run mad-spark evaluate "The AI dreamed of electric sheep"

# Compare ideas
uv run mad-spark compare "Business communication" -r "Email" -r "AI video"

# List evaluators
uv run mad-spark list-evaluators
```

### Python API

```python
import asyncio
from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator

async def run_analysis():
    orchestrator = SimpleQADIOrchestrator(temperature_override=1.2)
    result = await orchestrator.run_qadi_cycle(
        user_input="How can we reduce plastic waste?",
        context="Focus on practical, scalable solutions"
    )
    
    print(f"Core Question: {result.core_question}")
    print(f"\nBest Answer: {result.final_answer}")
    print(f"\nAction Plan:")
    for i, action in enumerate(result.action_plan):
        print(f"{i+1}. {action}")

asyncio.run(run_analysis())
```

For detailed API examples and advanced usage patterns, see [DEVELOPMENT.md](DEVELOPMENT.md).

## 🧬 Genetic Evolution

```bash
# Evolve ideas
uv run mad-spark evolve "How can we reduce food waste?"
uv run mad-spark evolve "Climate solutions" --quick
uv run mad-spark evolve "New product ideas" --generations 5 --population 20
```

### How It Works

1. **Initial**: QADI generates starting hypotheses
2. **Fitness**: 5-criteria scoring (Novelty 20%, Impact 30%, Cost 20%, Feasibility 20%, Risks 10%)
3. **Selection**: Best hypotheses breed
4. **Crossover**: Combine parent ideas (75% rate)
5. **Mutation**: Add variations (15% rate)
6. **Repeat**: Until optimal solutions emerge

**Features**: Parallel evaluation | Smart caching (50-70% reduction) | Checkpointing | Diversity preservation

## How QADI Works

1. **Q**: Extract core question ("I want X" → "What causes Y?")
2. **A**: Generate hypotheses (H1, H2, H3...)
3. **D**: Evaluate & determine best answer
4. **I**: Verify with real examples

**Temperature**: 0.0-0.5 (conservative) | 0.6-1.0 (balanced) | 1.1-2.0 (creative)

## Architecture

- **QADI Orchestrator**: 4-phase implementation
- **Unified Evaluator**: 5-criteria scoring
- **Evolution Engine**: AI-powered genetic algorithms with caching
- **Phase Optimization**: Optimal hyperparameters per phase

See [DEVELOPMENT.md](DEVELOPMENT.md) for details.

## Extension

- Implement `ThinkingAgentInterface` for custom agents
- Implement `EvaluatorInterface` for custom metrics
- Components auto-register on import

## Testing

```bash
# Unit tests (no API needed)
uv run pytest tests/ -m "not integration"

# Integration tests (requires API key)
uv run pytest tests/ -m integration

# All tests
uv run pytest
```

**Reliability**: Format validation | Mock-reality alignment | Graceful degradation

### CI Test Policy

**Update Tests For**: New features | Bug fixes | Parser changes | Integration changes

**Required Tests**: Smoke tests | Format validation | Regression tests | CLI validation

```bash
# Run before push
uv run pytest tests/ -m "not integration"
```

## Development

```bash
uv sync --dev
uv run pytest
uv run mypy src/
uv run black src/ tests/ && uv run isort src/ tests/
```

## Session Handover

### Last Updated: 2025-07-24 05:10

#### Recently Completed

- ✅ **PR #51**: Comprehensive System Testing and Validation Suite
  - CLI argument validation with helpful error messages for evolution parameters
  - 31 new tests: CLI validation (10), multi-perspective integration (12), system validation (9)  
  - Performance optimizations: O(N²) → O(N) hypothesis finding, DRY principle in CLI defaults
  - Fixed all PR review feedback: GeneticAlgorithm constructor, type annotations, code quality
  - Verified all README commands work correctly with real LLM calls

#### Next Priority Tasks

1. **Performance Optimization**: Profile multi-perspective analysis for cost efficiency
   - Context: Each perspective runs a full QADI cycle (4 phases), plus synthesis = 13 LLM calls for 3 perspectives
   - Approach: Implement caching, shared reasoning phases, or perspective relevance filtering

2. **User Experience Enhancement**: Add interactive perspective selection
   - Context: Users may want to see available perspectives before analysis
   - Approach: Add `--list-perspectives` flag and interactive selection mode

#### System Health

- **Test Coverage**: 31 new validation tests ensure system reliability
- **CLI Robustness**: Clear error messages prevent user confusion
- **Documentation**: All commands verified working with current system

#### Session Learnings
- **Best Practices Documented**: Key learnings from PR #51 on system testing and CLI development have been added to the [project style guide](CLAUDE.md#system-testing-best-practices-pr-51). Highlights include:
  - Verifying bot claims with file system checks.
  - Using built-in functions for performance and `parser.get_default()` for DRYer CLI code.
  - The importance of real-world, comprehensive testing.

## Documentation

- [DEVELOPMENT.md](DEVELOPMENT.md): Architecture, API reference, contribution guide
- [RESEARCH.md](RESEARCH.md): QADI methodology background
- [SESSIONS.md](SESSIONS.md): Development history

## License

MIT
