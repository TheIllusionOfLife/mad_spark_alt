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
uv run mad_spark_alt "Your question" --evolve --generations 2 --population 10 --diversity-method semantic --verbose
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
- **[SESSIONS.md](SESSIONS.md)** - Development history
- **[STRUCTURED_OUTPUT.md](docs/STRUCTURED_OUTPUT.md)** - Gemini structured output implementation
- **[docs/](docs/)** - Additional documentation including CLI usage, examples, and API reference

## Session Handover

### Last Updated: August 04, 2025 06:39 PM JST

#### Recently Completed
- ✅ **[PR #101] QADI Display Formatting Fix**: Improved hypothesis title extraction and analysis formatting (Aug 4, 2025)
  - **Title Extraction**: Fixed evaluation scores showing generic category labels instead of actual hypothesis content
  - **Analysis Section**: Added approach number display and fixed line break issues for numbered lists
  - **Code Quality**: Addressed all feedback from 4 bot reviewers (claude[bot], coderabbitai[bot], cursor[bot], gemini-code-assist[bot])
  - **Pattern Update**: Updated CLAUDE.md with critical fix - never use category-based extraction for titles
- ✅ **[PR #97] Phase 1 Performance Optimizations**: Batch semantic operators for 60-70% performance improvement (Aug 4, 2025)
  - **Batch Processing**: BatchSemanticCrossoverOperator and BatchSemanticMutationOperator implemented
  - **Performance**: Heavy workloads now complete 60-70% faster through batch LLM operations
  - **ID Mapping Consistency**: Established a systematic mapping pattern to resolve inconsistencies. The system uses 1-based indexing for user-facing prompts and test mocks (intuitive for humans: "pair 1", "pair 2"), while using 0-based indexing for internal parsing logic (efficient for arrays: index 0, index 1). This prevents systematic failures from ID system mismatches.
  - **Type Safety**: Resolved MyPy Optional[T] → List[T] issues with explicit null checks
  - **CI Robustness**: Systematic 4-layer fixes (logic → tests → feedback → types)
- ✅ **[PR #93] Semantic Diversity Calculation**: Implemented Gemini embeddings for true semantic understanding (Aug 4, 2025)
  - **API Integration**: Added embedding support to LLM provider with proper batch endpoint
  - **Dual Strategy**: Semantic (embedding-based) and Jaccard (word-based) diversity calculators
  - **Fallback Design**: Automatic fallback to Jaccard if API fails
  - **User Feedback**: Clear display of which diversity method is being used
  - **Cost Optimization**: Caching embeddings by content hash to minimize API calls
- ✅ **[PR #92] Fitness Score Unification**: Refactored `confidence_score` to `overall_fitness` throughout codebase (Aug 3, 2025)
  - **Consistency**: Unified naming across all evaluation and evolution systems
  - **Migration**: Proper dual-field support during transition period
- ✅ **[PR #91] Documentation Updates**: Verified development plan and organized legacy docs (Aug 3, 2025)
- ✅ **[PR #90] Documentation & Phase 2 Display**: Fixed reviewer feedback and improved formatting (Aug 3, 2025)
  - **Reviewer Coordination**: Addressed feedback from claude[bot], coderabbitai[bot], gemini-code-assist[bot]
  - **Documentation**: Updated title extraction docs to mention all punctuation marks (.?!)
  - **Display Format**: Consolidated Phase 2 entries in Session Handover
- ✅ **[PR #89] Structured Output & Display Format**: Comprehensive testing and documentation (Aug 3, 2025)
  - **Structured Output**: Created STRUCTURED_OUTPUT.md, added 453 lines of integration tests, fixed schema types, and verified Gemini API integration
  - **Display Format**: Improved hypothesis presentation by extracting first sentence as title and using simple numbered list, per user request
  - **Code Quality**: Addressed all feedback from multiple bot reviewers
- ✅ **[PR #87] UX Improvements**: Fixed output truncation, evaluation scores, and evolution display (Aug 2, 2025)
  - **Smart Truncation**: Implemented regex-based sentence boundary detection
  - **Evaluation Scores**: Added approach titles and consistent metric ordering
  - **Evolution Output**: Cleaned parent references from genetic algorithm output
- ✅ **[PR #85] Parallel Evolution Processing**: Eliminated sequential bottleneck with batch LLM operations (Aug 2, 2025)
  - **Critical Fix**: Heavy workloads now complete without timeout
  - **Performance**: 60-70% execution time reduction through parallel processing

#### Next Priority Tasks

*Note: This section is the primary source of truth for upcoming development work. Tasks listed here have been verified against the current codebase (August 2025).*

**📋 Implementation Plan**: See [IMPLEMENTATION_PLAN_DIVERSITY_BREAKTHROUGH.md](docs/IMPLEMENTATION_PLAN_DIVERSITY_BREAKTHROUGH.md) for detailed implementation roadmap.

1. **Performance Optimization: Diversity Calculation**
   - **Status**: Active Development Needed
   - **Issue**: O(n²) complexity in both `JaccardDiversityCalculator` (jaccard_diversity.py:44-64) and `GeminiDiversityCalculator` (gemini_diversity.py:79)
   - **Impact**: Severe performance degradation with large populations (nested loops comparing all pairs)
   - **Note**: Gemini uses fast numpy operations for cosine similarity, but still O(n²)
   - **Approach**:
     - Profile current implementations with large populations
     - Research optimized algorithms (MinHash, LSH, or sampling-based approaches)
     - Target O(n log n) or better complexity
     - Validate diversity metrics remain meaningful after optimization

2. **Directed Evolution Mode**: Implement targeted evolution strategies  
   - **Status**: Next Major Development Priority
   - **Context**: Build on batch processing foundation to add intelligent evolution stages
   - **Implementation**:
     - Add "directed evolution" where mutations target specific weaknesses
     - Implement different evolution stages: diversification → intensification → synthesis
     - Apply special "enhancement" mutations only to elite individuals
     - Use different temperature/creativity settings for elite vs general population

3. **Plugin Architecture**: Create plugin system for custom operators
   - **Status**: Extensibility Enhancement
   - **Context**: Make batch processing extensible for domain-specific operators
   - **Implementation**:
     - Abstract operator interfaces with registration system
     - Plugin discovery and loading mechanism
     - Documentation for creating custom operators

#### Completed Tasks ✅
- ✅ **Phase 1 Performance Optimizations** (PR #97): Batch LLM operations for 60-70% performance improvement. See 'Recently Completed' for details.
- **Performance Benchmarking Suite**: Already exists in `tests/performance_benchmarks.py`
- **Title Length Extension**: Already implemented - uses 150 characters with smart truncation (qadi_simple.py)
- **Evolution Progress**: Basic text indicators exist and function adequately
- **Semantic Diversity Calculator**: Implemented with Gemini embeddings (PR #93)

#### Known Issues / Blockers
- None currently blocking development
- **Performance baseline established**: Evolution system now reliably completes without timeouts

#### Session Learnings
- **Hypothesis Title Extraction (PR #101)**: Critical lessons for user-facing text extraction
  - **Category-Based Extraction Trap**: Never use keyword matching to classify content - it returns generic labels not actual content
  - **Actual Content Extraction**: Always extract the first meaningful sentence/phrase from the hypothesis itself
  - **Japanese/English Support**: Use language-specific punctuation patterns (.。!?！？) for sentence boundaries
  - **List Item Detection**: Use precise regex `r'(?<=[\s。.!?！？])\s*\((\d{1,2})\)\s*(?=[^\s\d])'` to avoid treating years as list items
  - **Systematic PR Review**: Even "simple" fixes benefit from 4-bot review process (found 4 distinct issues)
  - **Constants Over Magic Numbers**: Extract all numeric thresholds to named constants at module level
- **Phase 1 Performance Optimization Success**: Batch semantic operators deliver dramatic improvements
  - **Systematic Bug Fixing**: Layer-by-layer approach (logic → tests → feedback → types) prevents cascading issues
  - **ID Mapping Consistency**: Critical to establish 3-way consistency: prompts, parsing logic, test mocks
  - **Batch Processing Performance**: Replacing multiple sequential LLM calls with a single batch call provides dramatic performance gains (e.g., from O(n) to O(1) API calls for a generation's mutations)
  - **MyPy Optional Types**: Always use explicit null checks rather than type ignore comments for Optional[T] → List[T] operations
- **Semantic Diversity Implementation**: Successfully added Gemini embeddings for true semantic understanding of idea diversity
  - Critical API fix: Use `:batchEmbedContents` not `:embedContent` endpoint
  - Request format: Must use `json=payload` with proper batch structure
  - Cost awareness: Embeddings cost $0.0002 per 1K tokens (10x initial error)
  - Validation requirement: Always check response length matches request
- **User Feedback Importance**: Added clear display of which diversity method (Jaccard vs Semantic) is being used
  - Pattern: Similar to existing "[FALLBACK]" and evolution operators display
  - Implementation: Both CLI commands now show diversity method with guidance
- **PR Review Efficiency**: Systematic 4-phase protocol found all issues from 5 reviewers
  - gemini-code-assist[bot]: Critical API endpoint correction
  - cursor[bot]: Request handling standardization
  - coderabbitai[bot]: Code cleanup (unused imports, f-strings)
- **Previous Learnings Reinforced**:
  - Always test with real API to catch integration issues
  - Follow TDD strictly - wrote tests first, then implementation
  - Fallback patterns essential for reliability (Jaccard fallback for API failures)

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
- [ ] Implement cache warming strategies for semantic operators
- [ ] Add diversity calculation benchmarks to performance test suite

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
  
- ✅ **Phase 1 Performance Optimizations** - COMPLETED: Batch LLM operations for 60-70% performance improvement. See 'Session Handover' section for implementation details.

- [ ] **Directed Evolution Mode** (#4) - Add targeted evolution strategies
  - Add "directed evolution" where mutations target specific weaknesses
  - Implement different evolution stages: diversification → intensification → synthesis
  - Apply special "enhancement" mutations only to elite individuals
  - Use different temperature/creativity settings for elite vs general population

## License

MIT
