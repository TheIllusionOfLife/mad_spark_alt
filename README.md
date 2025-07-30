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

# Add genetic evolution to optimize ideas (uses semantic operators by default)
uv run mad_spark_alt "How can we reduce plastic waste?" --evolve

# Display help
uv run mad_spark_alt --help
```

### Advanced Options

```bash
# Temperature control (creativity level)
uv run mad_spark_alt "Your question" --temperature 1.2

# Customize evolution parameters (generations = 2, population = 5 by default)
uv run mad_spark_alt "Your question" --evolve --generations 3 --population 8

# Use traditional operators for faster evolution
uv run mad_spark_alt "Your question" --evolve --traditional
```

## Important: Two Different Commands

This project provides two distinct command-line interfaces:

### 1. `mad_spark_alt` (Main Command - Recommended)
- **Purpose**: QADI analysis with optional evolution
- **Usage**: `uv run mad_spark_alt "question" [--evolve]`
- **Evolution**: Added via `--evolve` flag
- **Defaults**: generations=2, population=5

### 2. `mad-spark` (Alternative CLI)
- **Purpose**: General CLI for various evaluation tasks
- **Usage**: `uv run mad-spark [command] [options]`
- **Evolution**: Separate `evolve` subcommand
- **Commands**: `evolve`, `evaluate`, `compare`, `batch-evaluate`

**Note**: Most users should use `mad_spark_alt` for QADI analysis. The `mad-spark` CLI is for advanced evaluation workflows.

## How QADI Works

1. **Q**: Extract core question
2. **A**: Generate hypotheses
3. **D**: Evaluate & determine best answer
4. **I**: Verify with real examples

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


## Documentation

- [DEVELOPMENT.md](DEVELOPMENT.md): Architecture, API reference, contribution guide
- [RESEARCH.md](RESEARCH.md): QADI methodology background
- [SESSIONS.md](SESSIONS.md): Development history
- [SEMANTIC_OPERATORS_IMPLEMENTATION.md](SEMANTIC_OPERATORS_IMPLEMENTATION.md): Semantic evolution operators guide

## Session Handover

### Last Updated: July 30, 2025 04:10 PM JST

#### Recently Completed
- ✅ [PR #67]: Improve qadi_simple evolution to match mad-spark evolve
  - Implemented dynamic hypothesis generation based on population parameter
  - Fixed hypothesis display to show all generated hypotheses
  - Enhanced timeout calculations for realistic LLM response times
  - Added comprehensive test coverage for evolution improvements
  
- ✅ [PR #66]: Remove quick mode and update checkpoint frequency
  - Simplified CLI by removing rarely-used quick mode
  - Updated default checkpoint frequency for better performance
  - Cleaned up related code and tests

#### Next Priority Tasks
1. **Extract nested functions to module level**
   - Source: [PR #67 review comments]
   - Context: `calculate_evolution_timeout` function defined inside `run_qadi_analysis`
   - Approach: Move to module level for better testability and reusability

2. **Implement placeholder tests**
   - Source: [tests/qadi_simple_evolution_test.py:92-113]
   - Context: Empty test methods create false coverage impression
   - Approach: Either implement meaningful tests or remove placeholders

3. **Refactor complex deduplication logic**
   - Source: [qadi_simple.py:459-537]
   - Context: Multiple fallback strategies make code hard to follow
   - Approach: Extract into helper functions like `deduplicate_by_similarity()`

4. **Performance benchmarks for diversity calculation**
   - Source: [README.md - Future Improvements]
   - Context: O(n²) complexity in population diversity calculations
   - Approach: Add benchmarks and consider optimization strategies

#### Known Issues / Blockers
- None currently blocking development

#### Session Learnings
- **CI Test Synchronization**: Tests must use exact implementation values to prevent divergence
- **User Display Pattern**: Show what users request with clarification when system uses less
- **Dynamic Parameter Propagation**: Successfully implemented CLI → orchestrator → prompts flow

## Future Improvements

### Performance Optimizations
- [ ] Add performance benchmarks for diversity calculation (O(n²) complexity)
- [ ] Implement cache warming strategies for semantic operators
- [ ] Consider async batch evaluation for better parallelism
- [ ] Optimize string similarity calculations with better algorithms

### Enhanced Features
- [ ] Add more sophisticated fitness estimation for unevaluated offspring
- [ ] Implement adaptive mutation strategies based on population convergence
- [ ] Add visualization tools for evolution progress
- [ ] Support for multi-objective optimization

### Code Quality
- [ ] Add more integration tests with real LLM calls
- [ ] Implement comprehensive error recovery for LLM API failures
- [ ] Add telemetry and monitoring for production usage
- [ ] Create plugin architecture for custom operators

### Documentation
- [ ] Add tutorial for implementing custom genetic operators
- [ ] Create performance tuning guide
- [ ] Document best practices for population/generation sizing
- [ ] Add troubleshooting guide for common issues

## License

MIT
