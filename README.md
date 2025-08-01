# Mad Spark Alt - AI-Powered QADI Analysis System

Intelligent analysis system using QADI methodology (Question → Abduction → Deduction → Induction) to provide structured, multi-perspective insights on any topic.

## Features

- **QADI Methodology**: Structured 4-phase analysis for any question or problem
- **Universal Evaluation**: Impact, Feasibility, Accessibility, Sustainability, Scalability
- **Multiple Analysis Modes**: Simple, hypothesis-driven, multi-perspective
- **Temperature Control**: Adjust creativity level (0.0-2.0)
- **Audience-Neutral**: Practical insights for everyone, not just businesses
- **Real-World Examples**: Concrete applications at individual, community, and systemic levels
- **Structured Output**: Utilizes Gemini's structured output API for reliable parsing of hypotheses and scores

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

### Last Updated: August 01, 2025 06:06 PM JST

#### Recently Completed
- ✅ [PR #76]: Complete comprehensive QADI evolution fixes - Address all 5 user-identified issues
  - Fixed missing line breaks in hypothesis display
  - Removed H-prefix from deduction analysis  
  - Implemented unified QADI 5-criteria scoring system
  - Fixed evolution result collection from all generations
  - Enhanced display with "High Score Approaches" and detailed scoring
- ✅ [PR #74]: Remove H+number prefix, suppress evolution logs, and improve structured output
- ✅ [PR #71]: Implement Gemini structured output for reliable parsing

#### Completed Priority Tasks
✅ **Remove Smart Selector**: Simplified evolution system
   - Removed SmartOperatorSelector class completely  
   - Replaced probabilistic selection with simple if-available logic
   - Evolution uses semantic operators when available and enabled via `use_semantic_operators`
   - Result: Cleaner codebase, better predictability, same functionality

✅ **Evaluation Context Enhancement**: Added scoring context to evolution
   - Created EvaluationContext dataclass with original question and target improvements
   - Semantic operators now receive evaluation context for targeted mutations/crossovers  
   - Enhanced prompts guide evolution toward specific fitness improvements
   - Result: More targeted evolution that improves weak scores

#### Next Priority Tasks
1. **Evolution Timeout Handling**: Improve timeout management
   - Source: User test output shows evolution timing out at 290s
   - Context: Population 10 + 3 generations exceeds timeout
   - Approach: Better progress tracking, earlier termination, or adaptive timeouts

#### Known Issues / Blockers
- Evolution with large populations (10+) and multiple generations (3+) times out
- Mutation responses occasionally appear truncated (token limit issue)

#### Session Learnings
- **Prompt-Response Consistency**: Critical to ensure LLM prompts match expected response format
- **Comprehensive Testing**: Always test the exact output format, not just functionality
- **Review Bot Thoroughness**: Modern PRs have multiple bot reviewers checking different aspects
- **Evolution Display**: Users expect to see results from all generations, not just final

## Technical Notes

### Structured Output Implementation

The system now uses Gemini's structured output feature (`responseMimeType` and `responseSchema`) to improve reliability of hypothesis generation and score parsing. This addresses previous issues with brittle regex-based parsing:

- **Hypothesis Generation**: Uses JSON schema to ensure consistent hypothesis extraction
- **Score Parsing**: Structured output for reliable extraction of evaluation scores
- **Evolution Operators**: Mutation and crossover operations use structured schemas
- **Fallback Mechanism**: Gracefully falls back to text parsing if structured output fails

This implementation significantly reduces "Failed to extract enough hypotheses" errors and ensures more reliable parsing of LLM responses.

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

### Evolution System Enhancements
- ✅ **Remove Smart Selector** - COMPLETED: Simplified by removing SmartOperatorSelector class entirely
  - ✅ Replaced probabilistic decisions with simple: if semantic operators available → use them
  - ✅ Kept `use_semantic_operators` as simple on/off switch
  - ✅ Result: Cleaner codebase, better predictability, same functionality
  
- ✅ **Evaluation Context** (#6) - COMPLETED: Pass scoring context to evolution
  - ✅ Added EvaluationContext dataclass with original question and target improvements
  - ✅ Semantic operators now receive evaluation context for targeted mutations/crossovers
  - ✅ Enhanced prompts guide evolution toward specific fitness improvements
  - ✅ Result: More targeted evolution that improves weak scores
  
- [ ] **Enhanced Semantic Operators** (#2) - Improve prompts for higher scores
  - Modify mutation prompts to explicitly target evaluation criteria
  - Add "score improvement" directive to semantic operators
  - Include evaluation criteria (impact, feasibility, etc.) in mutation/crossover context
  - Create "breakthrough" mutation type that explicitly aims for higher scores
  
- [ ] **Directed Evolution Mode** (#4) - Add targeted evolution strategies
  - Add "directed evolution" where mutations target specific weaknesses
  - Implement different evolution stages: diversification → intensification → synthesis
  - Apply special "enhancement" mutations only to elite individuals
  - Use different temperature/creativity settings for elite vs general population


## License

MIT
