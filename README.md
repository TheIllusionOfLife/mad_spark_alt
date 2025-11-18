# Mad Spark Alt - AI-Powered QADI Analysis System

Intelligent analysis system using QADI methodology (Question → Abduction → Deduction → Induction) to provide structured, multi-perspective insights on any topic.

## Features

- **QADI Methodology**: Structured 4-phase analysis for any question or problem
- **Multi-Provider LLM Support**: Choose between Gemini API (cloud) or Ollama (local/free)
- **Universal Evaluation**: Impact, Feasibility, Accessibility, Sustainability, Scalability
- **Multiple Analysis Modes**: Simple, hypothesis-driven, multi-perspective
- **Temperature Control**: Adjust creativity level (0.0-2.0)
- **Audience-Neutral**: Practical insights for everyone, not just businesses
- **Real-World Examples**: Concrete applications at individual, community, and systemic levels
- **Structured Output**: Utilizes Gemini's structured output API for reliable parsing of hypotheses and scores
- **Multimodal Support**: Analyze images (both providers), PDFs and URLs (Gemini only) alongside text

### Multimodal Capabilities (New!)

Mad Spark Alt now supports multimodal analysis. Both Gemini and Ollama support image analysis, while PDFs and URLs require Gemini:

**Supported Input Types:**
- **Images**: PNG, JPEG, WebP, HEIC (up to 20MB per image) - **Both Gemini and Ollama**
- **Documents**: PDF files with vision understanding (up to 1000 pages) - **Gemini only**
- **URLs**: Fetch and analyze web content (up to 20 URLs per request) - **Gemini only**

**Usage via Python API:**
```python
import asyncio
from mad_spark_alt.core.llm_provider import GoogleProvider, LLMRequest
from mad_spark_alt.core.multimodal import MultimodalInput, MultimodalInputType, MultimodalSourceType

async def analyze_image():
    provider = GoogleProvider(api_key="your-key")

    # Analyze an image
    image_input = MultimodalInput(
        input_type=MultimodalInputType.IMAGE,
        source_type=MultimodalSourceType.FILE_PATH,
        data="path/to/image.png",
        mime_type="image/png"
    )

    request = LLMRequest(
        user_prompt="Describe this architecture diagram",
        multimodal_inputs=[image_input]
    )

    response = await provider.generate(request)
    print(response.content)  # AI description of the image
    print(f"Images processed: {response.total_images_processed}")

    await provider.close()

# Run the async function
asyncio.run(analyze_image())
```

**Example Use Cases:**
- Analyze system architecture diagrams for improvement suggestions
- Process research papers (PDF) to extract key findings
- Compare product screenshots for competitive analysis
- Fetch and synthesize information from multiple web sources
- Mixed-modal: Combine images, documents, and URLs in one analysis

**Cost**: Images/pages add ~258 tokens each. See [Cost Information](#cost-information) below.

## Installation

```bash
git clone https://github.com/TheIllusionOfLife/mad_spark_alt.git
cd mad_spark_alt
uv sync  # Or: pip install -e .
```

## Quick Start

```bash
# Setup API key (REQUIRED for Gemini, optional for Ollama-only usage)
echo "GOOGLE_API_KEY=your_key_here" > .env
# Note: Skip this step if using --provider ollama exclusively
```

### Basic Usage (Short Alias)

```bash
# Simple QADI analysis (default command)
msa "How can we reduce plastic waste?"

# With genetic evolution
msa "How can we reduce plastic waste?" --evolve

# Display help
msa --help
```

### Using with uv run

```bash
# If msa alias doesn't work, use full command
uv run msa "How can we reduce plastic waste?"
uv run msa "Your question" --evolve
```

### LLM Provider Selection

Choose between cloud API (Gemini) or local inference (Ollama):

```bash
# Auto mode (default): Prefers Ollama, falls back to Gemini
# - Text/images → Ollama (free local, with Gemini fallback if Ollama unavailable)
# - Documents/URLs → Gemini (required for processing)
msa "Your question" --provider auto

# Force Gemini API (cloud, costs money)
msa "Your question" --provider gemini

# Force Ollama (free local, requires Ollama running)
msa "Your question" --provider ollama

# Ollama setup (one-time)
# 1. Install: https://ollama.ai
# 2. Run: ollama serve
# 3. Pull model: ollama pull gemma3:12b-it-qat
# 4. Use: msa "Your question" --provider ollama
```

**Provider Comparison:**

| Provider | Cost | Speed | Multimodal | Notes |
|----------|------|-------|------------|-------|
| Gemini | ~$0.01/query | Fast (10s) | ✅ Images, PDFs, URLs, TXT, CSV, JSON, MD | Requires API key |
| Ollama | Free | Slower (20-30s) | ✅ Images only | Requires local setup |
| Auto | Mixed | Variable | ✅ Full support (hybrid mode for docs) | Ollama-first with fallback |

**Current Limitations:**
- **Subcommand provider selection** - `evaluate`, `batch-evaluate`, and `compare` subcommands do not support provider selection yet (they use their own evaluator registry, not QADI orchestration)

**Recent Improvements:**
- ✅ **Hybrid routing** - When using `--document` or `--url` flags with auto mode, Gemini extracts content once, then Ollama runs QADI phases locally (cost optimization)
- ✅ **Multi-format documents** - Support for PDF, TXT, CSV, JSON, and Markdown files (text files read directly without API calls)
- ✅ **URL security** - SSRF prevention blocks internal URLs, private IPs, and cloud metadata endpoints
- ✅ **Content caching** - SHA256-based caching prevents redundant file reads (1-hour TTL)
- ✅ **Temperature control** - Users now have full control over temperature settings without automatic clamping
- ✅ **SDK/API fallback** - SDK users now have automatic fallback via `ProviderRouter.run_qadi_with_fallback()`
- ✅ **Main command provider selection** - `msa --provider` flag allows explicit provider selection for QADI analysis

### Advanced Options

```bash
# Temperature control (creativity level)
msa "Your question" --temperature 1.2

# Customize evolution parameters (generations = 2, population = 5 by default)
msa "Your question" --evolve --generations 3 --population 8

# Use traditional operators for faster evolution
msa "Your question" --evolve --traditional

# Use semantic diversity calculation for enhanced idea variety
msa "Your question" --temperature 2.0 --evolve --generations 2 --population 10 --diversity-method semantic --verbose

# Analyze an image with QADI (works with both Gemini and Ollama)
msa "Analyze this design for improvement" --image design.png

# Process documents (PDF, TXT, CSV, JSON, MD supported)
msa "Summarize key findings" --document research.pdf
msa "Analyze this data" --document data.csv --document notes.md

# Combine multiple modalities (requires Gemini for PDFs/URLs)
msa "Compare these approaches" --image chart1.png --image chart2.png --url https://example.com/article

# Multiple documents and URLs (hybrid mode: Gemini extracts, Ollama runs QADI)
msa "Synthesize insights" --document report.pdf --document analysis.txt --url https://source.com
```

### Export Results

Save analysis results to files for later review, sharing, or integration with other tools:

```bash
# Export QADI analysis to JSON (default format)
msa "How can we improve remote work?" --output results.json

# Export to Markdown format
msa "How can we improve remote work?" --output report.md --format md

# Export QADI + Evolution results
msa "Create sustainable urban transportation" --evolve --output analysis.json

# Export with explicit format selection
msa "Your question" --output results.json --format json

# Evolution results include both QADI analysis and genetic algorithm outputs
msa "Design innovative recycling systems" --evolve --generations 3 --population 8 --output full_analysis.json
```

**Export Formats:**

- **JSON**: Machine-readable, includes all metadata and scores
  - QADI sections: core_question, hypotheses, hypothesis_scores, final_answer, action_plan, verification_examples, verification_conclusion
  - Evolution sections: best_ideas, fitness_progression, evolution_metrics
  - Metadata: LLM costs, processing statistics, timestamps

- **Markdown**: Human-readable, formatted for documentation
  - Headers and sections for easy navigation
  - Score tables with visual formatting
  - Action plans as numbered lists
  - Evolution results with fitness progression tables

**Output Structure Example (JSON with Evolution):**
```json
{
  "qadi_analysis": {
    "core_question": "...",
    "hypotheses": ["H1", "H2", "H3"],
    "hypothesis_scores": [...],
    "final_answer": "...",
    "action_plan": [...],
    "verification_examples": [...],
    "verification_conclusion": "...",
    "metadata": {...}
  },
  "evolution_results": {
    "best_ideas": ["Evolved idea 1", "Evolved idea 2"],
    "total_generations": 3,
    "execution_time": 120.5,
    "fitness_progression": [
      {"generation": 0, "best_fitness": 0.75, "avg_fitness": 0.68},
      {"generation": 1, "best_fitness": 0.82, "avg_fitness": 0.74}
    ],
    "evolution_metrics": {...}
  },
  "exported_at": "2025-11-09T09:28:17.550850+00:00"
}
```

## Command Reference

The main command is **`msa`** (short for Mad Spark Alt), which provides QADI analysis by default:

```bash
# Default: QADI analysis
msa "Your question here"

# List available commands
msa --help
```

### Subcommands

While QADI analysis is the default (no subcommand needed), additional commands are available:

```bash
# List available evaluators
msa list-evaluators

# Evaluate text with specific evaluators
msa evaluate "text" --evaluators diversity_evaluator

# Use multiple evaluators
msa evaluate "text" --evaluators diversity_evaluator,quality_evaluator

# Use all evaluators (default)
msa evaluate "text"
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
# Instead of: msa "prompt" --evolve
# Use: scripts/run_nohup.sh "prompt" --evolve

# Example
scripts/run_nohup.sh "Create a game concept" --evolve --generations 3 --population 10
```

Output will be saved to `outputs/msa_output_TIMESTAMP.txt`.

See the `scripts/run_nohup.sh` script for our solution to terminal timeout issues.

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete system architecture, data flows, and technical standards (single source of truth)
- **[CLAUDE.md](CLAUDE.md)** - AI assistant instructions and development patterns
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Architecture, API reference, contribution guide
- **[RESEARCH.md](RESEARCH.md)** - QADI methodology background
- **[session_handover.md](session_handover.md)** - Development history and learnings
- **[STRUCTURED_OUTPUT.md](docs/STRUCTURED_OUTPUT.md)** - Gemini structured output implementation
- **[docs/](docs/)** - Additional documentation including CLI usage, examples, and API reference

## Development Roadmap

### Current Priorities

**Active Development:**
1. **Performance Optimization: Diversity Calculation** - Reduce O(n²) complexity to enable larger population sizes
2. **Directed Evolution Mode** - Intelligent evolution with targeted mutations and multi-stage strategies

**Recently Completed:**
- ✅ **Pydantic Validation Migration - COMPLETE** (PR #141, #142) - All 6 phases implemented with 962 tests passing
- ✅ **Result Export & Persistence** (PR #130) - JSON/Markdown export with security validation
- ✅ **Unified CLI Architecture** (PR #126) - Single `msa` command with -4,164 lines reduction
- ✅ **Phase 1 Performance Optimizations** (PR #97) - 60-70% execution time improvement through batch operations
- ✅ **Semantic Diversity Calculator** (PR #93) - Gemini embeddings for true semantic understanding

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
- **[session_handover.md](session_handover.md)** - Development progress, learnings, and future roadmap
- **[DEPRECATED.md](DEPRECATED.md)** - Deprecated features and migration notes
- **[docs/](docs/)** - API reference, CLI usage, code examples, and detailed guides

## Session Status

**Last Updated**: November 15, 2025 04:05 PM JST

**Recent Highlights**:
- ✅ PR #142: Complete Pydantic Validation Migration (Phases 3b, 4, 5, 6) - 962/962 tests passing
- ✅ PR #141: Pydantic Schema Foundation (Phases 1, 2, 3a) - Multi-provider compatibility
- ✅ Addressed all reviewer feedback from 4 bot reviewers (8 actionable issues fixed)
- ✅ 3-Layer graceful fallback pattern implemented (Pydantic → JSON → Text)
- ✅ Real API integration tests passing (8/8 tests with Google Gemini)
- ✅ All CI checks passing, system stable

**Migration Complete**: Pydantic validation now used across all QADI phases and evolution operators with comprehensive test coverage and backward compatibility.

**Next Priorities**:
- Monitor production Pydantic validation performance
- Performance Optimization: Diversity Calculation (O(n²) → O(n))
- Consider adding OpenAI/Anthropic integration tests

For complete session details, see **[session_handover.md](session_handover.md)**.

## License

MIT
