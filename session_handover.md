# Session Handover

## Last Updated: December 26, 2025

---

## Current Work In Progress

**None** - Clean state, ready for new work.

---

## Recently Completed

### PR #176: Upgrade Embedding Model - MERGED ✅ (December 26, 2025)
**Summary**: Migrated from deprecated `text-embedding-004` to `gemini-embedding-001`.

**Key Changes**:
- ✅ Default model: `text-embedding-004` → `gemini-embedding-001`
- ✅ Updated pricing: $0.15 per 1M tokens ($0.00015/1K)
- ✅ Default 768 dimensions (MRL-supported, can reduce to 256)
- ✅ Updated all test files with new model name

### PR #175: Disable Gemini Fallback for Explicit Provider - MERGED ✅ (December 26, 2025)
**Summary**: When `--provider ollama` is explicitly set, disable all fallback mechanisms.

**Key Changes**:
- ✅ `--provider auto`: Falls back to Gemini (unchanged)
- ✅ `--provider ollama`: No fallback, fails if Ollama unavailable
- ✅ `--provider ollama --diversity-method semantic`: Error at startup (requires Gemini embeddings)

### PR #173: Gemini 3 Flash + Model Registry - MERGED ✅ (December 18, 2025)
**Summary**: Upgraded to Gemini 3 Flash with centralized model registry for future multi-provider support.

**Key Changes**:
- ✅ Model registry (`core/model_registry.py`) with immutable `ModelSpec` dataclass
- ✅ Default model: `gemini-2.5-flash` → `gemini-3-flash-preview`
- ✅ Updated pricing: $0.50/$3.00 per million tokens (input/output)
- ✅ Token multiplier support for reasoning overhead (3x for Gemini 3)
- ✅ Increased deduction `max_tokens`: 3000 → 8000 (thinking mode consumes output tokens)

### PR #172: Architecture Docs Sync - MERGED ✅ (December 17, 2025)
**Summary**: Updated documentation to reflect December 2025 capabilities.

### PR #171: Dependency Cleanup - MERGED ✅ (December 17, 2025)
**Summary**: Removed unused FastAPI/SQLAlchemy dependencies (7 packages total).

### PR #167: Increase Evolution Timeouts - MERGED ✅ (December 16, 2025)
**Summary**: Extended timeouts for Ollama semantic operators to allow completion of evolution runs with max parameters.

**Key Changes**:
- ✅ `OLLAMA_INFERENCE_TIMEOUT`: 180s → 600s (10 min per request)
- ✅ `CLI_MAX_TIMEOUT_SECONDS`: 900s → 3000s (50 min total)
- ✅ Updated tests and documentation
- ✅ Verified with max params: `--generations 5 --population 10` (1043s, 31% fitness gain)

### PR #166: Redesign Induction Phase - MERGED ✅ (December 15, 2025)
**Summary**: Replaced disconnected induction output with rich synthesis connecting all QADI phases.

**Key Changes**:
- ✅ Rich synthesis output with recommended approach, supporting evidence, and action items
- ✅ Fixed JSON schema format for Ollama (lowercase types, not Google uppercase)
- ✅ Added parse failure visibility and hypothesis/score length validation


### PR #165: Outlines for Ollama Structured Output - MERGED ✅ (December 15, 2025)
**Summary**: Added Outlines library for constrained grammar generation at token level.

**Key Changes**:
- ✅ `inline_schema_defs()` utility to expand `$ref/$defs` for Ollama compatibility
- ✅ Multimodal bypass: Outlines flattens to plain text, use native API for images
- ✅ Timeout protection via `asyncio.wait_for()` with `OLLAMA_INFERENCE_TIMEOUT`
- ✅ AsyncClient caching to avoid resource leaks (`provider.close()` required)
- ✅ `Optional[T]` incompatibility workaround: use `Field(default="")` instead

### PR #161: CLI Output Display Improvements - MERGED ✅ (December 7, 2025)
**Summary**: Improved CLI output with explicit QADI phases and better readability.

**Key Changes**:
- ✅ Explicit QADI phase labels (Q-Question, A-Abduction, D-Deduction, I-Induction)
- ✅ Rich Table for score comparison
- ✅ Full hypotheses display (not truncated)
- ✅ Connected action plan to recommended approach
- ✅ Suppressed INFO logs (WARNING level default)
- ✅ Fixed `'str' object has no attribute 'get'` in Ollama responses

**Pattern Added**: "Structured Output Over Prompt Engineering" rule in CLAUDE.md
- Never request formatting (numbers, bullets) in prompts when using structured output
- Use Pydantic schema constraints instead

### PR #160: Japanese UAT Fixes - MERGED ✅ (November 20, 2025)
- Fixed URL processing (disable structured output when URLs present)
- Fixed JSON Unicode export (`ensure_ascii=False`)
- Documented Ollama language mirroring limitation

### PR #158: Evaluate Flag Mode - MERGED ✅ (November 19, 2025)
- Converted subcommand to `--evaluate` flag
- Flag guarding validation pattern
- Markdown format support

---

## Deferred Items

### LOW PRIORITY
1. **Performance Benchmark Test Flakiness**
   - `test_real_ollama_performance_benchmark` is flaky (38s > 30s threshold)
   - Solution: Increase threshold to 60s or skip in CI
   - Files: `tests/test_ollama_provider.py:394`

---

## Known Issues / Blockers

- **DNS Rebinding**: URL validation doesn't resolve hostnames - documented in CLAUDE.md
- **CSV/JSON Duplication**: Formatting logic duplicated between CLI and provider_router - documented
- **Ollama Language**: Doesn't respect language mirroring - use `--provider gemini` for non-English

---

## Code Quality Status

- **Total Tests**: 1112 passing
- **Type Checking**: mypy clean
- **CI/CD**: All checks passing

---

## Previous Sessions Summary

| Date | PRs | Key Work |
|------|-----|----------|
| Dec 26, 2025 | #175, #176 | Explicit provider fallback control, embedding model upgrade |
| Dec 18, 2025 | #171, #172, #173 | Gemini 3 Flash, model registry, docs sync, cleanup |
| Dec 16, 2025 | #166, #167 | Induction redesign, evolution timeouts |
| Dec 15, 2025 | #165 | Outlines for Ollama structured output |
| Dec 7, 2025 | #161 | CLI output improvements, QADI phases |
| Nov 20, 2025 | #160 | Japanese UAT fixes |
| Nov 18-19, 2025 | #154, #157, #158 | Ollama paths, CLI syntax, evaluate flag |
| Nov 18, 2025 | #151 | Hybrid routing polish, 29 tests |
| Nov 16, 2025 | #148, #149 | Hybrid routing, centralized fallback |
| Nov 15-16, 2025 | #144, #146, #147 | Multi-provider, documentation |
| Nov 14-15, 2025 | #141, #142 | Pydantic migration |
