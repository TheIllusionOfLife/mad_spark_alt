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

### Last Updated: August 01, 2025 12:57 PM JST

#### Recently Completed
- ✅ [PR #74]: Fix hypothesis format and evolution issues
  - **UX Improvement**: Removed redundant H+number prefix from hypothesis display
  - **Clean Output**: Suppressed evolution debug logs by using logger.debug instead of logger.warning
  - **Enhanced Reliability**: Added structured output support to single mutation operator
  - **Bot Review Integration**: Systematically addressed feedback from claude[bot], coderabbitai[bot], and gemini-code-assist[bot]
  - **Comprehensive Testing**: Added 5 new test files with 17+ tests covering all changes
  - **Type Safety**: Fixed mypy type checking errors in semantic_operators.py

- ✅ [PR #71]: Implement Gemini structured output for reliable parsing
  - **Major Reliability Improvement**: Eliminated "Failed to extract enough hypotheses" errors
  - **Structured Output API**: Added Gemini `responseMimeType` and `responseSchema` support
  - **Graceful Fallbacks**: JSON parsing → regex parsing → default values for robustness
  - **Critical Bug Fix**: Resolved ID mapping misalignment in evolution operators (cursor bot detection)
  - **Comprehensive Testing**: 25+ tests covering edge cases (0-based, non-sequential IDs)
  - **Bot Integration**: Systematic processing of automated review feedback

#### Next Priority Tasks
1. **Extract nested functions to module level**
   - Source: [PR #67 review comments]
   - Context: `calculate_evolution_timeout` function defined inside `run_qadi_analysis`
   - Approach: Move to module level for better testability and reusability

2. **Performance benchmarks for diversity calculation**
   - Source: [README.md - Future Improvements]
   - Context: O(n²) complexity in population diversity calculations
   - Approach: Add benchmarks and consider optimization strategies

3. **Refactor duplicated crossover fallback logic**
   - Source: [PR #74 gemini-code-assist[bot] review]
   - Context: `_generate_crossover_fallback` has duplicated if/else structure
   - Approach: Extract fallback templates to dictionary, DRY principle

4. **Consider optional structured output enhancements**
   - Source: [PR #71 claude[bot] review comments]
   - Context: Schema validation improvements and configurable token limits
   - Approach: Low priority - current implementation is robust

#### Known Issues / Blockers
- None currently blocking development

#### Session Learnings
- **User Experience Focus**: Small changes like removing H+number prefix significantly improve output clarity
- **Logging Level Discipline**: Use logger.debug for internal messages, logger.warning only for user-visible issues
- **Systematic Bot Review**: Different bots provide feedback via PR comments, PR reviews, and line comments - check all three
- **Incremental Structured Output**: Can add JSON schemas to individual operators without full system refactor
- **Type Safety in Evolution**: Optional types need explicit handling to prevent mypy errors with LLM responses

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
- [ ] **Remove Smart Selector** - Simplify by removing SmartOperatorSelector class entirely
  - Replace probabilistic decisions with simple: if semantic operators available → use them
  - Keep `use_semantic_operators` as simple on/off switch
  - Rationale: Current threshold (0.9) means we essentially always want semantic operators
  
- [ ] **Evaluation Context** (#6) - Pass scoring context to evolution
  - Pass original user question + current best score to evolution operators
  - Let semantic operators know what scores they need to beat
  - Enable targeted improvements based on current performance
  
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

#### Next Priority Tasks
1. **Remove Smart Selector**: Simplify evolution system
   - Source: README.md TODO
   - Context: Current threshold (0.9) means we always use semantic operators anyway
   - Approach: Replace probabilistic selection with simple if-available logic

2. **Evaluation Context Enhancement**: Pass scoring context to evolution
   - Source: README.md TODO #6
   - Context: Evolution operators need to know what scores to beat
   - Approach: Pass original question + current best score to operators

3. **Evolution Timeout Handling**: Improve timeout management
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

## License

MIT
