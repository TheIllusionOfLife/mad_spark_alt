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
- **Purpose**: QADI analysis for idea generation
- **Example**: `uv run mad_spark_alt "prompt"`
- **Commands**: `uv run mad_spark_alt --help`

### 2. `mad-spark` (Alternative CLI)
- **Purpose**: General CLI for various tasks
- **Example**: `uv run mad-spark [command] "prompt"`
- **Commands**: `uv run mad-spark --help`

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


## Known Issues

### 2-Minute Timeout in Some Environments

When running long commands (especially with `--evolve`), you may encounter a timeout after exactly 2 minutes:
```text
Command timed out after 2m 0.0s
```

This is caused by the execution environment (terminal/shell/IDE), not the application itself.

**Solution**: Use the provided nohup wrapper script for long-running tasks:
```bash
# Instead of: uv run mad_spark_alt "prompt" --evolve
# Use: ./run_nohup.sh "prompt" --evolve

# Example
./run_nohup.sh "Create a game concept" --evolve --generations 3 --population 10
```

Output will be saved to `outputs/mad_spark_alt_output_TIMESTAMP.txt`. 

See [EVOLUTION_TIMEOUT_FIX.md](EVOLUTION_TIMEOUT_FIX.md) for detailed information.

## Documentation

- [DEVELOPMENT.md](DEVELOPMENT.md): Architecture, API reference, contribution guide
- [RESEARCH.md](RESEARCH.md): QADI methodology background
- [SESSIONS.md](SESSIONS.md): Development history
- [SEMANTIC_OPERATORS_IMPLEMENTATION.md](SEMANTIC_OPERATORS_IMPLEMENTATION.md): Semantic evolution operators guide
- [EVOLUTION_TIMEOUT_FIX.md](EVOLUTION_TIMEOUT_FIX.md): Timeout issue workaround

## Session Handover

### Last Updated: July 31, 2025 08:45 PM JST

#### Recently Completed
- ✅ [PR #69]: Fix timeout workaround script issues
  - Solved 2-minute terminal timeout issue with nohup wrapper
  - Added environment variable loading and unbuffered output
  - Comprehensive test coverage for all script functionality
  - Addressed all bot review feedback with proper fixes

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
- **Terminal Timeout Fix**: 2-minute timeout comes from terminal/shell environment, not Python. Only nohup successfully bypasses it.
- **Integration Test Pattern**: Tests must verify actual behavior (file creation), not just stdout assertions.
- **PR Review Efficiency**: Bot reviews come from 3 sources (PR comments, reviews, line comments). Batch similar fixes.
- **Bash Script Robustness**: Always use `set -euo pipefail` and handle `cd` failures explicitly.
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
