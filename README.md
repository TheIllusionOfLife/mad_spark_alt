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

# Use semantic diversity calculation for enhanced idea variety (slower but more accurate)
uv run mad_spark_alt "Your question" --temperature 2.0 --evolve --generations 2 --population 10 --diversity-method semantic --verbose
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

#### Evaluator Filtering

The `mad-spark evaluate` command supports filtering which evaluators to run:

```bash
# List available evaluators
uv run mad-spark list-evaluators

# Use a specific evaluator
uv run mad-spark evaluate "text" --evaluators diversity_evaluator

# Use multiple evaluators
uv run mad-spark evaluate "text" --evaluators diversity_evaluator,quality_evaluator

# Use all evaluators (default)
uv run mad-spark evaluate "text"
```

## How QADI Works

1. **Q**: Extract core question
2. **A**: Generate hypotheses
3. **D**: Evaluate & determine best answer
4. **I**: Verify with real examples

## Diversity Calculation Methods

The evolution system uses diversity calculation to prevent premature convergence and maintain idea variety throughout generations. Two methods are available:

### Jaccard Diversity (Default)
- **Speed**: Fast, word-based similarity calculation
- **Method**: Compares ideas using Jaccard similarity on word sets
- **Best for**: Quick evolution runs, development, testing
- **Usage**: `--diversity-method jaccard` (default)

### Semantic Diversity 
- **Speed**: Slower, requires Gemini API calls for embeddings
- **Method**: Uses text-embedding-004 model to create 768-dimensional semantic vectors
- **Accuracy**: More precise understanding of conceptual similarity vs surface-level word overlap
- **Best for**: Production runs where idea quality and semantic variety are priorities
- **Usage**: `--diversity-method semantic`
- **Requirements**: GOOGLE_API_KEY for embedding generation

**Example Comparison:**
- Jaccard: "reduce plastic waste" vs "decrease plastic pollution" = different (50% word overlap)
- Semantic: Same concepts = highly similar (0.85+ cosine similarity)

**Recommendation**: Use Jaccard for development and quick testing, Semantic for final production runs where conceptual diversity matters most.

## Cost Information

### LLM API Pricing (Gemini 2.5 Flash)
Based on official Google Cloud pricing (as of August 2025):
- **Input**: $0.30 per million tokens
- **Output**: $2.50 per million tokens  
- **Embeddings**: $0.20 per million tokens (text-embedding-004)

### Cost Simulation: Evolution Run

For the heaviest evolution setting with `--population 10 --generations 5` (maximum allowed):

| Phase | Operation | Estimated Cost | Actual Cost* |
|-------|-----------|----------------|--------------|
| **QADI Processing** | 4 LLM calls (Q→A→D→I) | $0.012 | Included |
| **Evolution** | 5 generations × 5 calls/gen | $0.050 | Included |
| **Fitness Evaluation** | Initial + 5 generations | $0.050 | Included |
| **Diversity (Semantic)** | 6 embedding calls | $0.001 | Included |
| **Total** | ~36 API calls | **$0.11** | **$0.016** |

*Actual cost from real run with semantic diversity, verbose output, and maximum settings. The significant difference is due to caching (14% hit rate), batch operations, and efficient token usage.

### Cost Optimization

1. **Batch Operations**: Already implemented - saves 10+ LLM calls per run
2. **Caching**: Fitness evaluations and embeddings are cached by content
3. **Jaccard Diversity**: Free alternative to semantic embeddings
4. **Smaller Populations**: Use `--population 5` to halve evolution costs

### Performance vs Cost Trade-offs

| Configuration | Time | Cost | Quality | Usage |
|--------------|------|------|---------|--------|
| Basic QADI only | ~10s | $0.002 | Good baseline | Quick exploration |
| Evolution (pop=3, gen=2) | ~60s | $0.005 | Better diversity | Typical usage |
| Evolution (pop=5, gen=3) | ~180s | $0.008 | Great results | Extended run |
| Evolution (pop=10, gen=5) | ~450s | $0.016 | Maximum quality | Heavy/research |
| With semantic diversity | +30s | +$0.001 | Conceptual diversity | When needed |

**Note**: Actual costs may vary based on prompt length and response verbosity.

## Architecture

- **QADI Orchestrator**: 4-phase implementation
- **Unified Evaluator**: 5-criteria scoring
- **Evolution Engine**: AI-powered genetic algorithms with caching
- **Diversity Calculation**: Multiple methods (Jaccard word-based, Semantic embedding-based)
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

See the `run_nohup.sh` script for our solution to terminal timeout issues.

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete system architecture, data flows, and technical standards (single source of truth)
- **[CLAUDE.md](CLAUDE.md)** - AI assistant instructions and development patterns
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Architecture, API reference, contribution guide
- **[RESEARCH.md](RESEARCH.md)** - QADI methodology background
- **[SESSION_HANDOVER.md](SESSION_HANDOVER.md)** - Development history and learnings
- **[STRUCTURED_OUTPUT.md](docs/STRUCTURED_OUTPUT.md)** - Gemini structured output implementation
- **[docs/](docs/)** - Additional documentation including CLI usage, examples, and API reference

## Development Roadmap

### Current Priorities

**Active Development:**
1. **Result Export & Persistence** - Add `--output` flag to main `mad_spark_alt` command for saving analysis results
2. **Performance Optimization: Diversity Calculation** - Reduce O(n²) complexity to enable larger population sizes
3. **Directed Evolution Mode** - Intelligent evolution with targeted mutations and multi-stage strategies

**Recently Completed:**
- ✅ **Phase 1 Performance Optimizations** (PR #97) - 60-70% execution time improvement through batch operations
- ✅ **Semantic Diversity Calculator** (PR #93) - Gemini embeddings for true semantic understanding
- ✅ **Structured Output Implementation** (PR #71) - Reliable parsing with Gemini's responseSchema

For detailed development history, completed tasks, and session learnings, see **[SESSION_HANDOVER.md](SESSION_HANDOVER.md)**.

## Technical Notes

### Structured Output Implementation

The system now uses Gemini's structured output feature (`responseMimeType` and `responseSchema`) to improve reliability of hypothesis generation and score parsing. This addresses previous issues with brittle regex-based parsing:

- **Hypothesis Generation**: Uses JSON schema to ensure consistent hypothesis extraction
- **Score Parsing**: Structured output for reliable extraction of evaluation scores
- **Evolution Operators**: Mutation and crossover operations use structured schemas
- **Fallback Mechanism**: Gracefully falls back to text parsing if structured output fails

This implementation significantly reduces "Failed to extract enough hypotheses" errors and ensures more reliable parsing of LLM responses.

## Documentation Map

- **[README.md](README.md)** (you are here) - Quick start, installation, and overview
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design, components, and technical architecture
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Development setup, contribution guide, and testing
- **[RESEARCH.md](RESEARCH.md)** - QADI methodology and academic background
- **[CLAUDE.md](CLAUDE.md)** - AI assistant development patterns and learned practices
- **[SESSION_HANDOVER.md](SESSION_HANDOVER.md)** - Development progress, learnings, and future roadmap
- **[DEPRECATED.md](DEPRECATED.md)** - Deprecated features and migration notes
- **[docs/](docs/)** - API reference, CLI usage, code examples, and detailed guides

## Session Handover

### Last Updated: November 08, 2025 11:46 AM JST

#### Recently Completed
- ✅ **[PR #119]**: Split semantic_operators.py (Step 13) - **Phase 3 Item 1/4 Complete**
  - 1,926 → 62 lines (97% reduction!)
  - Created 4 focused modules: semantic_utils (210), operator_cache (200), semantic_mutation (903), semantic_crossover (656)
  - Added 26 baseline integration tests, all 779 tests passing
  - Addressed 9 review issues across 3 cycles (gemini-code-assist, coderabbitai×2)
  - Cache keys now include full context with deterministic MD5 hashing

- ✅ **[PR #118]**: UnifiedQADIOrchestrator MultiPerspective strategy (Step 11b)
- ✅ **[PR #117]**: UnifiedQADIOrchestrator Simple strategy (Step 11a)
- ✅ **[PR #115]**: MultiPerspective Refactoring (507→312 lines, -38%)
  - Phase 2 now **100% complete** (6/6 items)

- ✅ **[PR #114]**: Removed legacy orchestrators (738 lines)
  - Deleted enhanced_orchestrator.py, robust_orchestrator.py, fast_orchestrator.py
  - Updated examples to use SmartQADIOrchestrator
  - Phase 2 Item 10 complete

- ✅ **[PR #112]**: BaseOrchestrator infrastructure implementation (Step 7)
  - Created abstract base class with circuit breaker pattern
  - Added shared orchestration logic (context building, error handling)
  - 468 lines, 49 comprehensive tests passing

- ✅ **[PR #111]**: Phase logic extraction (Step 6)
  - Extracted all 4 QADI phases to standalone module (791 lines)
  - Reduced SimpleQADI from 1,003 → 221 lines
  - Clear separation of orchestration from phase logic

#### Refactoring Plan Progress (79% Complete)
**Reference**: See `refactoring_plan_20251106.md` for detailed plan

**Phase 1**: ✅ **100% Complete** (4/4 items)
- All quick wins implemented and merged

**Phase 2**: ✅ **100% Complete** (6/6 items)
- ✅ Items 5-8: Parsing utils, phase logic, base orchestrator, SimpleQADI refactored
- ✅ Item 9: MultiPerspective refactored (507 → 312 lines, 38% reduction)
- ✅ Item 10: Legacy orchestrators removed (738 lines)

**Phase 3**: ⏳ **In Progress** (1/4 items, 25%)
- ✅ Item 13: semantic_operators.py split (1,926 → 62 lines, 97% reduction!) - **COMPLETED 2025-11-08**
- ❌ Items 11-12, 14: Unified orchestrator, config system, deprecations - **TODO**

#### Next Priority Tasks

1. **[NEXT - Phase 3] Create Unified Orchestrator (Step 11)**
   - **Source**: refactoring_plan_20251106.md lines 854-919
   - **Context**: Single orchestrator with config-based behavior selection
   - **Dependencies**: Phase 2 complete ✅
   - **Estimate**: 3 days
   - **Note**: Phase 2 is now complete, ready to start Phase 3

2. **[Phase 3] Extract Config System (Step 12)**
   - **Source**: refactoring_plan_20251106.md
   - **Context**: Centralized configuration management
   - **Depends on**: Step 11 complete
   - **Estimate**: 2 days

3. **[Phase 3] Deprecate Old Orchestrators (Step 14)**
   - **Source**: refactoring_plan_20251106.md
   - **Context**: Add deprecation warnings to orchestrators replaced by Unified Orchestrator
   - **Depends on**: Step 11 complete
   - **Estimate**: 1 day

#### Known Issues / Blockers
- None - Phase 2 complete, ready for Phase 3

#### Session Learnings

**PR #119: Multi-Cycle Review Process (2025-11-08)**:
- Addressed 9 issues across 3 review cycles (gemini-code-assist, coderabbitai×2)
- Systematic priority handling: HIGH (cache TTL) → MEDIUM (DRY violations, imports, constants) → NITPICK (deterministic hashing) → DOCS (status clarity)
- All 3 sources extracted for every reviewer: comments, review bodies, line comments
- Real API testing validated refactoring preserved functionality
- **Lesson**: Multiple review cycles are normal for large refactorings - address feedback systematically by priority

**Semantic Operators Refactoring (2025-11-08)**:
- Split 1,926-line monolithic file into 4 focused modules (97% reduction in main file)
- TDD approach: 26 baseline tests before refactoring, all 779 tests passing after
- 100% backward compatibility via re-export module pattern
- Cache key improvements: full context inclusion + deterministic MD5 hashing
- **Lesson**: Re-export modules enable aggressive refactoring while maintaining compatibility

**Refactoring Delegation Pattern (2025-11-07 to 2025-11-08)**:
- MultiPerspective refactoring achieved 38% code reduction (507→312 lines)
- TDD approach: baseline tests first, then refactor, then update tests for new architecture
- Delegation pattern: `_run_perspective_analysis()` creates SimpleQADI instances per perspective
- Removed 195 lines of duplicate QADI phase logic by delegating to SimpleQADI
- Real API testing validated no regressions (timeouts, truncation, errors)
- **Lesson**: Write comprehensive tests before refactoring to catch regressions early

**Phase 2 Completion (2025-11-07 to 2025-11-08)**:
- All 6 Phase 2 items complete: parsing utils, phase logic, base orchestrator, SimpleQADI, MultiPerspective, legacy removal
- Total lines removed in Phase 2: ~2,600+ lines across 6 PRs
- Architecture now follows clear patterns: shared infrastructure (base), phase execution (phase_logic), orchestration (orchestrators)
- **Lesson**: Consistent patterns across refactoring make each subsequent step easier

## License

MIT
