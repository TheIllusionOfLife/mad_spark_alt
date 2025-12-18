# CLAUDE.md

> **For project documentation:** [README.md](README.md) | [ARCHITECTURE.md](ARCHITECTURE.md) | [DEVELOPMENT.md](DEVELOPMENT.md) | [RESEARCH.md](RESEARCH.md)

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Mad Spark Alt is a Multi-Agent Idea Generation System powered by LLMs using the QADI methodology (Question → Abduction → Deduction → Induction).

**Current State**: Advanced LLM-powered system with smart orchestration, dynamic prompt engineering, multi-provider support (OpenAI, Anthropic, Google), and cost-aware processing.

## ⚠️ CRITICAL: Never Use Template Agents

**Template agents produce generic, meaningless responses.**

- ❌ NEVER use `qadi_working.py` or template-only implementations
- ✅ ALWAYS use LLM-powered tools (Google API, etc.)
- ✅ Use `msa "question"` for analysis (unified CLI)

## Architecture Overview

### Core Components

1. **LLM-Only Agent System** - Template agents are meaningless, always use LLM implementations
2. **Smart Orchestration** (`core/smart_orchestrator.py`) - Intelligent agent selection with cost tracking
3. **LLM Integration** (`core/llm_provider.py`) - Gemini API with robust JSON parsing and retry logic
4. **Registry Pattern** (`core/registry.py`) - Dynamic agent registration at runtime
5. **Dynamic Prompt Engineering** (`core/prompt_classifier.py`) - Auto-detects question types with 100% accuracy

### Key Patterns

1. **JSON Parsing**: LLMs return markdown-wrapped JSON; use `safe_json_parse()` with fallbacks
2. **Cost Tracking**: Centralized in `cost_utils.py`; use `calculate_llm_cost_from_config()` directly
3. **Error Handling**: Phase failures don't crash cycle; comprehensive logging throughout
4. **Async Operations**: All operations are async; use `asyncio.run()` for sync contexts

### CI Test Update Policy

**CRITICAL**: These changes REQUIRE CI test updates:
- **Parser/Format Changes**: Format validation tests with realistic data
- **New Features**: Smoke tests, CLI tests if applicable
- **Bug Fixes**: Regression tests preventing recurrence
- **Integration Changes**: Mock updates reflecting real response formats

**Validation**: `uv run pytest tests/ -m "not integration"` before push

### CI/CD Pipeline
- **Optimized**: Single Python 3.11, essential checks only (~2m23s)
- **Integration Tests**: Excluded (require API keys), marked `@pytest.mark.integration`
- **Local Testing**: Run mypy and pytest before push

## Important Notes

1. **Always use `uv run`** prefix for CLI commands
2. **LLM agents require GOOGLE_API_KEY**
3. **Cost tracking is automatic** - check `result.llm_cost`
4. **JSON parsing is critical** - handle markdown-wrapped responses

## Project-Specific Patterns

### LLM Score Parsing Reliability (CRITICAL)
- **Issue**: Mock-Reality Divergence - mocks use `Novelty: 0.8` but real LLMs return `* Novelty: 0.8 - explanation`
- **Prevention**: Integration tests with real LLM calls validate prompt-parser compatibility
- **Parser**: Must handle markdown bold, bullet points, explanatory text
- **Token Limits**: Deduction needs 1500+ tokens for complete analysis

### Reasoning Model Token Overhead (Gemini 3+)
- **Issue**: Thinking/reasoning models consume output tokens for internal reasoning, causing truncation
- **Fix**: Increase `max_tokens` significantly (e.g., 3000 → 8000) with token multiplier (3x for Gemini 3)
- **Registry**: Use `model_registry.get_model_spec()` for model-specific token multipliers

### Prompt-Response Format Consistency (CRITICAL)
- **Issue**: Prompt criteria must EXACTLY match response format examples
- **Fix**: `Impact: [score] - [explanation]` in both description and format sections
- **Testing**: Full prompt-to-response cycle with real LLM catches mismatches

### Structured Output Over Prompt Engineering (CRITICAL)
- **NEVER** request formatting (numbers, bullets) in prompts when using structured output
- **DO** use Pydantic schema field descriptions to guide content
- **DO** use `min_length`/`max_length` constraints in Pydantic fields
- **Rule**: Prompts describe WHAT content to provide; schemas define HOW to structure it
- **Anti-pattern**: Prompt `"1. [First item]"` + Schema `List[str]` → LLM includes "1." → double numbering on display
- **Correct**: Prompt `"Provide 3 items"` + Schema `List[str] = Field(min_length=3)` → clean output
- **Never** use regex to "fix" LLM formatting - fix the prompt/schema instead

### Type Safety Requirements
- **Pattern**: `field if field is not None else default_value` for Optional fields
- **CI Failure**: mypy errors common; run `uv run mypy src/` before push

### Registry Architecture
- **Global**: `agent_registry` and `evaluator_registry`
- **Clear Before Tests**: `agent_registry.clear()` in test setup
- **CLI**: Must call `load_env_file()` before registry use

### QADI Orchestration
- **Phase Order**: Question → Abduction → Deduction → Induction
- **Parallel**: Use `run_parallel_generation` for efficiency
- **Context**: Each phase builds on previous results

### Import Requirements
- **No Inline Imports**: Module level only for CI
- **Relative**: `from ...core import X` pattern
- **Type Checking**: `from typing import TYPE_CHECKING` for circular deps

### Logging Levels
- **User-Visible**: `logger.warning()` or higher for issues users need to know
- **Internal Debug**: `logger.debug()` for fallbacks, parsing failures, internal state

### Deprecation Best Practices
- **Warn at module import**: Use `warnings.warn()` at module level
- **Provide migration path**: Specify the new module/function to use
- **Use `stacklevel=2`**: Pinpoint the caller location in warnings

### Evolution System
- **API**: `GeneticAlgorithm()` with `EvolutionRequest` object, access `result.final_population`
- **Mutation**: Creates new objects even at 0% rate; use 0.3+ for diversity
- **Deduplication**: Required to avoid identical ideas; use 85% similarity threshold
- **Testing**: Variance tolerance (`final_fitness >= initial_fitness * 0.9`), not exact values
- **Config Validation**: Generations 2-5, Population 2-10, tournament/parallel ≤ population_size

### Semantic Evolution Operators
- **Smart Selection**: Population diversity and fitness determine semantic vs traditional
- **Batch Operations**: `mutate_batch()` for 60-70% execution time reduction
- **Caching**: Prevents redundant LLM calls; Jaccard similarity for diversity
- **AsyncMock Pattern**: Use `new=AsyncMock(return_value=...)` not `new_callable=AsyncMock`

### Multi-Perspective QADI
- **Auto-Detection**: Environmental, Personal, Technical, Business, Scientific, Philosophical
- **Score Parsing**: `criteria_mappings` MUST match `HypothesisScore` fields; use named arguments
- **Parallel Analysis**: `asyncio.gather()` for concurrent perspectives
- **Relevance**: Primary=1.0, subsequent=`0.8 - (index * 0.1)`

### CLI Development
- **Argument Validation**: Helpful error messages with suggestions
- **DRY Defaults**: `parser.get_default()` not hardcoding
- **Terminal Detection**: `sys.stdout.isatty()` before Rich formatting

### PR Review Bot Patterns
- **Sources**: Check PR comments, PR reviews, AND line comments
- **Priorities**: Critical (security) > High (APIs) > Medium (quality) > Low (style)
- **Script Robustness**: Always `set -euo pipefail` in bash scripts

### Complex Integration Testing (PR #56 Learnings)
- **Problem**: 55% fix commits indicate insufficient initial test coverage
- **Solution**: Write integration tests alongside unit tests from start
- **Realistic Data**: Production-like responses in ALL mocks (copy actual API responses)
- **Config Matrix**: Test ALL parameter combinations and boundaries

### Claude Code Custom Commands
- **$ARGUMENTS Support**: Custom commands accept arguments via `$ARGUMENTS`
- **Example**: `/fix_pr_since_commit 1916eed` passes "1916eed" as `$ARGUMENTS`
- **Implementation**: Use `${ARGUMENTS:-default_value}` for optional parameters

## Known Limitations

### Hybrid Routing Security (PR #151)
1. **DNS Rebinding**: URL validation checks hostnames but does NOT resolve DNS. Hostnames that resolve to private IPs (DNS rebinding attacks) are not blocked. For maximum security in production, consider using a DNS resolver to verify the resolved IP is not private before fetching.

2. **CSV Parsing**: Simple line-based parsing (`content.strip().split("\n")`) may not handle quoted values with embedded newlines or non-comma delimiters. Consider using Python's `csv` module for more robust parsing in future enhancements.

3. **Token Estimation**: Uses rough approximation (1 token ≈ 4 characters). This varies significantly between content types (code vs prose, different languages). Current warning thresholds provide sufficient buffer.

4. **PDF Caching**: Cache stores text file content but not PDF extraction results. Same PDF processed twice calls Gemini API both times. This is intentional (PDF extraction prompts may evolve), but costs more.

5. **CSV/JSON Formatting Duplication**: CSV and JSON formatting logic exists in both `unified_cli.py` (non-hybrid path) and `provider_router.py._read_text_document()` (hybrid path). This creates maintenance burden where future changes must be applied to both locations. Refactoring to share a single helper would require significant architectural changes (CLI is synchronous, provider_router is async). Current implementation keeps behavior consistent across both paths with test coverage.

### Gemini API Limitations (Japanese UAT Fixes - GitHub PR #160)

**Note**: These limitations were discovered during Japanese User Acceptance Testing and documented as part of the planned "PR #2" in the Japanese UAT issue sequence (not to be confused with GitHub PR numbers).

6. **URL Context + Structured Output Incompatibility**: Gemini API does not support using the `url_context` tool with structured output (`responseMimeType` + `responseJsonSchema`). When URLs are present in a request, the system automatically disables structured output and falls back to text parsing. This ensures URL processing works correctly but may reduce parsing reliability for complex JSON responses.

7. **Ollama Language Mirroring**: Ollama models (tested with gemma3:12b) do not reliably respect language mirroring instructions. Japanese input may produce English output. For non-English languages, use `--provider gemini` which handles language mirroring correctly.

### Outlines/Ollama Structured Output (PR #165)

8. **Token Estimation**: Outlines library doesn't provide token counts. Uses rough approximation (1 token ≈ 4 characters) for usage tracking. Actual token usage may vary.

9. **Pydantic Model Requirement**: Outlines path requires Pydantic model classes as `response_schema`. Dict-based JSON schemas use the fallback path (native Ollama API with `format` parameter).

10. **$defs Inlining**: Ollama has issues with `$defs`/`$ref` JSON Schema features (issues #8444, #8462). Schemas are pre-processed with `inline_schema_defs()` to expand all references inline. This increases schema size but ensures compatibility.

11. **AsyncClient Caching**: Ollama AsyncClient and Outlines models are cached per-provider instance to avoid resource leaks. Call `provider.close()` when done to release resources.

12. **Optional[T] Incompatibility**: Ollama's structured output has issues with `Optional`/`anyOf` patterns in JSON schemas. Use non-optional fields with default values instead (e.g., `str = Field(default="")` instead of `Optional[str]`).

13. **Multimodal Input Bypass**: Outlines is automatically skipped for requests with multimodal inputs (images). Outlines flattens prompts to plain text, losing image context. The native Ollama API properly handles images via the messages format with `images` field.

14. **Timeout Protection**: Outlines calls have timeout protection (`CONSTANTS.TIMEOUTS.OLLAMA_INFERENCE_TIMEOUT`, default 600s). Raises `LLMError` with `ErrorType.TIMEOUT` if the call takes too long.

## Completed Features Reference

| PR | Feature | Key Pattern |
|----|---------|-------------|
| #173 | Model Registry | `ModelSpec` dataclass; token multiplier for reasoning models; `get_model_spec()` API |
| #71, #107 | Structured Output | `responseJsonSchema` (NOT `responseSchema`); JSON→regex→default fallback |
| #141, #142 | Pydantic Schemas | `Field(ge=0.0, le=1.0)` for validation; 3-layer fallback (Pydantic→JSON→text) |
| #89, #101 | Phase 2 Display | First sentence as title; avoid category-based extraction |
| #69 | Timeout Workarounds | Use `nohup` for terminal-independent execution |
| #97 | Batch Processing | 1-based prompt → 0-based parsing → 1-based test mock pattern |
| #115 | Orchestrator Delegation | Delegate to simpler orchestrator; 38% code reduction |
| #126 | Unified CLI | `msa "question"` default; `--evolve` as flag; manual dispatch for optional args |
| #130 | Result Export | `to_dict()` internal, `to_export_dict()` user-facing; path traversal protection |
| #139 | System Constants | `CONSTANTS.CATEGORY.NAME` pattern; frozen dataclasses for immutability |
| #144 | Multi-Provider | `ProviderRouter` auto-selection; `finally` block for resource cleanup; enum type safety |
| #149 | Hybrid Routing | Gemini preprocess → Ollama QADI; fail-fast on empty inputs; `_is_ollama_connection_error()` helper |

**Detailed Documentation**:
- Pydantic: [MULTI_PROVIDER_SCHEMAS.md](docs/MULTI_PROVIDER_SCHEMAS.md)
- Structured Output: [STRUCTURED_OUTPUT.md](docs/STRUCTURED_OUTPUT.md)
- CLI Migration: [CLI_MIGRATION.md](docs/CLI_MIGRATION.md)

### Resource Management (PR #144)
```python
try:
    result = await orchestrator.run_qadi_cycle(question)
finally:
    if ollama_provider is not None:
        await ollama_provider.close()  # MUST cleanup aiohttp session
```

### 3-Layer Graceful Fallback (PR #142)
```python
try:
    result = Model.model_validate_json(content)  # Layer 1: Pydantic
except (ValidationError, json.JSONDecodeError):
    try:
        data = json.loads(content)  # Layer 2: Manual JSON
    except json.JSONDecodeError:
        result = extract_from_text(content)  # Layer 3: Regex
```

### Hybrid Routing with Fail-Fast (PR #149)
```python
# Preprocess with Gemini, run QADI with Ollama
extracted_context, cost = await router.extract_document_content(docs, urls)
if not extracted_context.strip():
    raise ValueError("No valid content extracted")  # Fail fast, don't mislead user
enhanced_input = f"Context:\n{extracted_context}\n\nQuestion: {user_input}"
result = await ollama_orchestrator.run_qadi_cycle(enhanced_input)
```
