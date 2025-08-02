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

**Recent Performance Improvements**: The parallel processing architecture (implemented in PR #85) significantly reduces execution time for heavy workloads through batch LLM operations. Tests show 60-70% performance improvement over sequential processing.

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
- [STRUCTURED_OUTPUT.md](docs/STRUCTURED_OUTPUT.md): Gemini structured output implementation

## Session Handover

### Last Updated: August 03, 2025 12:33 AM JST

#### Recently Completed
- ✅ **[PR #89] Structured Output Implementation**: Comprehensive testing and documentation (Aug 3, 2025)
  - **Documentation**: Created STRUCTURED_OUTPUT.md with complete implementation guide
  - **Testing**: Added 453 lines of integration tests with real LLM validation
  - **Schema Fixes**: Fixed INTEGER vs STRING type mismatch in mutation schema
  - **Code Quality**: Addressed all feedback from claude[bot], coderabbitai[bot], gemini-code-assist[bot]
  - **Integration**: Verified structured output works reliably with Gemini API
- ✅ **Phase 2 Display Format**: Improved hypothesis presentation in verbose mode (Aug 3, 2025)
  - **User Request**: Changed from "**Approach X:**" labels to simple numbered list (1., 2., 3.)
  - **Smart Formatting**: Extract first sentence as title, display description on next line
  - **Clean Output**: Remove duplicate numbering and approach prefixes from LLM responses
- ✅ **[PR #87] UX Improvements**: Fixed output truncation, evaluation scores, and evolution display (Aug 2, 2025)
  - **Smart Truncation**: Implemented regex-based sentence boundary detection
  - **Evaluation Scores**: Added approach titles and consistent metric ordering
  - **Evolution Output**: Cleaned parent references from genetic algorithm output
- ✅ **[PR #85] Parallel Evolution Processing**: Eliminated sequential bottleneck with batch LLM operations (Aug 2, 2025)
  - **Critical Fix**: Heavy workloads now complete without timeout
  - **Performance**: 60-70% execution time reduction through parallel processing

#### Next Priority Tasks
1. **Performance Benchmarking for Diversity Calculation**
   - Source: TODO in roadmap - O(n²) complexity issue
   - Context: Diversity calculation scales poorly with large populations
   - Approach: Profile current implementation, explore optimized algorithms

2. **Performance Benchmarking Suite**
   - Source: PR #85 + PR #83 combined achieve 60-70% + 46% improvements
   - Context: Need to ensure performance gains are consistent across all scenarios
   - Approach: 
     - Create automated benchmark suite testing various population/generation/operator combinations
     - Track execution time, memory usage, and cost metrics
     - Establish baseline performance targets

3. **Extend Title Length Limits**
   - Source: User testing showed 60-character limit too restrictive
   - Context: Meaningful approach titles need more space
   - Approach: Increase to 120-150 characters with smart truncation at phrase boundaries

4. **Evolution Progress Visualization**
   - Source: User feedback on understanding evolution progress
   - Context: Current text-based progress indicators are minimal
   - Approach: Add visual progress bars and generation summaries

#### Known Issues / Blockers
- None currently blocking development
- **Performance baseline established**: Evolution system now reliably completes without timeouts

#### Session Learnings
- **LLM Response Reliability**: Prompt-based formatting is fundamentally unreliable - structured output APIs are essential
- **Systematic PR Review**: Following 4-phase protocol caught all reviewer feedback across multiple sources
- **DRY in Practice**: Helper functions at module level prevent duplication and improve maintainability
- **Regex for Smart Truncation**: Pattern `r'[.!?]["\']?(?=\s|$)'` effectively finds sentence boundaries
- **UX Testing Insights**: Real user testing revealed issues unit tests missed:
  - 60-char title limits too restrictive for meaningful content
  - ANSI codes must be cleaned before any text processing
  - Evolution output needs post-processing to hide algorithm internals
- **Integration Test Importance**: Mock tests can hide format divergence - real LLM tests essential

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
- [x] **Parallel Evolution Processing**: Implemented batch LLM processing for genetic operations (dramatically reduces heavy workload execution time)
- [x] **Batch Semantic Operators**: Single LLM call processes multiple mutations simultaneously instead of sequential processing
- [ ] Add performance benchmarks for diversity calculation (O(n²) complexity)
- [ ] Implement cache warming strategies for semantic operators
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
  
- ✅ **Enhanced Semantic Operators** (#2) - COMPLETED: Improved prompts for higher scores
  - ✅ Modified mutation prompts to explicitly target evaluation criteria
  - ✅ Added "score improvement" directive to semantic operators
  - ✅ Included evaluation criteria (impact, feasibility, etc.) in mutation/crossover context
  - ✅ Created "breakthrough" mutation type for high-scoring ideas (fitness >= 0.8)
  - ✅ Added revolutionary mutation types: paradigm_shift, system_integration, scale_amplification, future_forward
  - ✅ Higher temperature (0.95) and token limits for breakthrough mutations
  - ✅ Result: High-performing ideas get revolutionary variations while regular ideas get standard semantic mutations
  
- [ ] **Directed Evolution Mode** (#4) - Add targeted evolution strategies
  - Add "directed evolution" where mutations target specific weaknesses
  - Implement different evolution stages: diversification → intensification → synthesis
  - Apply special "enhancement" mutations only to elite individuals
  - Use different temperature/creativity settings for elite vs general population


## License

MIT
