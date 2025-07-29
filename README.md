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

# Customize evolution parameters
uv run mad_spark_alt "Your question" --evolve --generations 3 --population 3
```

### Advanced Options

```bash
# Temperature control (creativity level)
uv run python qadi_simple.py "Your question" --temperature 1.2

# Use traditional operators for faster evolution
uv run python qadi_simple.py "Your question" --evolve --traditional
```

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

### Last Updated: July 29, 2025 06:19 PM JST

#### Recently Completed
- ✅ [PR #62]: Fix evolution display and hypothesis parsing issues
  - Enhanced QADI prompts for detailed 100+ word responses
  - Robust ANSI code removal with multiple targeted regex patterns
  - Fixed evolution config validation constraints
  - Added comprehensive test coverage for parsing edge cases
- ✅ [PR #61]: Remove batch evaluation functionality
- ✅ [PR #60]: Comprehensive performance optimizations and timeout fixes

#### Next Priority Tasks
1. **Expand Real LLM Integration Tests**: Improve coverage of integration tests with actual API calls
   - Source: README.md roadmap
   - Context: Current integration tests are minimal; need comprehensive real-world validation
   - Approach: Expand test suite with API key requirement, mark as integration tests
   
2. **Enhance LLM API Error Recovery**: Add more advanced error handling
   - Source: README.md roadmap
   - Context: Current retry logic is basic; need more robust failure handling like provider fallbacks
   - Approach: Implement fallback strategies and more granular error handling

3. **Fix Mock-Reality Divergence**: Ensure all test mocks match actual LLM formats
   - Source: PR #62 experience
   - Context: Tests passed but real usage failed due to format differences
   - Approach: Copy real API responses for all mock data

#### Known Issues / Blockers
- Evolution config has strict validation rules (generations: 2-5, population: 2-10)
- ANSI code patterns vary between LLM providers
- Test timeouts may occur with slow API responses

#### Session Learnings
- LLM output parsing requires multiple targeted regex patterns, not overly broad patterns like `.*` that may remove legitimate content
- Evolution display was showing summaries due to insufficient token limits (now 2500)
- Mock test data must exactly match real LLM response formats
- CI test updates are mandatory for parser changes and bug fixes

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
