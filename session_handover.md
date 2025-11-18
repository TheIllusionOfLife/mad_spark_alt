# Session Handover

## Last Updated: November 18, 2025 11:19 AM JST

---

## Current Work In Progress

None - All PRs merged, main branch clean.

---

## Follow-Up PRs Needed

### LOW PRIORITY

1. **Performance Benchmark Test Flakiness**
   - **Problem**: `test_real_ollama_performance_benchmark` is flaky (38s > 30s threshold)
   - **Solution**: Increase threshold to 60s or skip in CI
   - **Files**: `tests/test_ollama_provider.py:394`

---

## Recently Completed

### PR #151: Hybrid Routing Polish - MERGED ✅ (November 18, 2025)
**11 Commits with 1343 additions, 53 deletions across 6 files**:

**Key Features Implemented**:
- ✅ **Security**: SSRF prevention with comprehensive URL validation (schemes, private IPs, cloud metadata, percent-encoding bypasses)
- ✅ **Format Support**: TXT, CSV, JSON, MD documents processed directly without API calls (cost optimization)
- ✅ **Content Caching**: SHA256-based cache with mtime tracking prevents redundant processing
- ✅ **Size Limits**: Token estimation and content truncation prevents prompt overflow
- ✅ **CLI Help**: Updated help text to document hybrid mode behavior
- ✅ **29 Comprehensive Tests**: Full coverage for security, caching, formats, size limits

**Critical Bug Fixes During Review**:
1. **Cache Type Mismatch** (CRITICAL): `float(mtime_hash)` vs `int` comparison → precision loss → cache always invalidated
   - Fixed by keeping hash as `int` throughout lifecycle
2. **Inline Imports**: Violated CLAUDE.md pattern, moved `unquote` to module-level
3. **Cache Order Sensitivity**: Same documents in different order caused cache miss
   - Fixed by sorting `doc_paths` before mtime hash computation
4. **JSON ASCII Encoding**: Non-ASCII characters escaped to `\uXXXX`, bloating context
   - Fixed with `ensure_ascii=False`
5. **Blocking I/O in Async Path**: File reads blocked event loop
   - Fixed with `loop.run_in_executor()` for non-blocking I/O

**Technical Debt Addressed**:
- CSV/JSON formatting duplication documented as known limitation (architectural constraint)
- All 10 code review issues addressed (9 fixed, 1 documented)

**Test Results**: 29/29 passing, mypy clean, all CI checks passing

---

### PR #149: Hybrid Routing & Temperature Control - MERGED ✅
**8 Commits with 989 lines of new code**:

1. **68bcf5c** - Remove temperature clamping for better UX
2. **67198a9** - Implement hybrid routing (Gemini preprocess → Ollama QADI)
3. **abd5e07** - Update docs and tests (14 new tests)
4. **0363fda** - Address reviewer feedback (Ollama session leak fix)
5. **7aba86a** - DRY refactor (`_is_ollama_connection_error()` helper)
6. **f2b509b** - Fail fast on empty inputs
7. **3d17867** - Add regression tests (non-Ollama errors, empty extraction)
8. **aa2c929** - Type safety fixes (`str(path).lower()` vs `path.lower()`)

**Key Features**:
- ✅ Hybrid routing: Gemini extracts PDFs/URLs once → Ollama runs QADI locally (free)
- ✅ Temperature control: Users have full control, no automatic capping
- ✅ Fail-fast validation: Error when all documents invalid (no misleading output)
- ✅ 42 new tests for comprehensive coverage
- ✅ Cost breakdown: Preprocessing vs QADI costs tracked separately

### PR #148: Centralize Fallback Logic - MERGED ✅
- SDK-friendly `ProviderRouter.run_qadi_with_fallback()`
- Removed ~50 lines of duplication from CLI

### PR #147: Address PR #144 Reviewer Feedback - MERGED ✅
- Fixed evolution provider switching bug
- Thread-safe resource management
- Targeted error detection (keyword-based)

### PR #146: Reduce CLAUDE.md - MERGED ✅
### PR #145: Session Handover - MERGED ✅

---

## Session Learnings

### Cache Type Consistency (CRITICAL)
- **Problem**: Type mismatch between cache set/get causes silent invalidation
- **Root Cause**: Storing `float(hash_value)` but comparing against `int` from `hash()` → precision loss
- **Solution**: Keep hash values as integers throughout cache lifecycle
- **Pattern**: Use `Union[float, int]` type hint if field serves dual purpose
- **Added to**: `~/.claude/core-patterns.md` - Database & Caching section

### Async I/O in LLM Pipelines
- **Problem**: Blocking file I/O in async extraction paths blocks event loop
- **Solution**: Offload synchronous I/O to thread pool with `run_in_executor()`
- **Pattern**: Wrap blocking operations in typed inner function, await executor
- **Why**: Large document processing shouldn't block LLM API calls
- **Added to**: `~/.claude/domain-patterns.md` - LLM Integration Patterns

### Comprehensive Code Review Workflow
- **Pattern**: Systematic review → prioritize by severity → fix in batches → verify all
- **Critical**: Check ALL feedback sources (PR comments, PR reviews, line comments, CI annotations)
- **Efficiency**: Group related fixes into coherent commits (security, performance, quality)
- **Verification**: Run full test suite + mypy after each batch

### Inline Imports Are Anti-Patterns
- **Problem**: Violates CLAUDE.md "No Inline Imports" rule, affects CI
- **Solution**: Always import at module level, deduplicate if used multiple times
- **Why**: Better for static analysis, cleaner code structure

### Hybrid Routing Architecture
- **Pattern**: Preprocess with capable provider (Gemini), run reasoning with cost-effective provider (Ollama)
- **Benefit**: Single extraction avoids reprocessing in each QADI phase
- **Critical**: Fail-fast when no valid content extracted (don't mislead with empty context)

### DRY Helper Methods for Error Detection
- **Problem**: Duplicated Ollama error detection logic in multiple methods
- **Solution**: Extract to `_is_ollama_connection_error()` helper
- **Pattern**: `isinstance(error, (ConnectionError, OSError, TimeoutError)) or keyword_match`

### Type Safety with Path Objects
- **Problem**: `doc_path.lower()` fails if doc_path is Path object
- **Solution**: Always use `str(doc_path).lower()` for compatibility
- **Why**: Parameters may receive either Path or str objects

### Fail-Fast on Invalid Inputs
- **Problem**: Silently proceeding with empty extraction misleads users
- **Solution**: Raise ValueError when all documents invalid
- **Pattern**: Validate before expensive operations, provide clear error messages

### SDK-Friendly API Design
- **Pattern**: Use `Optional[tuple] = None` with normalization `tuple = tuple or ()`
- **Benefit**: SDK callers don't need to pass empty tuples explicitly

---

## Known Issues / Blockers

- **CI Performance**: Test job takes 6-7 minutes (acceptable)
- **DNS Rebinding**: URL validation doesn't resolve hostnames - documented limitation in CLAUDE.md
- **CSV/JSON Duplication**: Formatting logic duplicated between CLI and provider_router - documented as technical debt

---

## Code Quality Status

- **Total Tests**: 1075+ passing (29 new in PR #151)
- **Type Checking**: mypy clean (80 source files)
- **CI/CD**: All checks passing (test, build, claude-review, CodeRabbit)
- **Documentation**: README, CLAUDE.md, session_handover.md, global patterns updated
- **Coverage**: Comprehensive edge cases (SSRF attacks, cache invalidation, async I/O, size limits)

---

## Technical Debt Addressed

1. **Temperature Capping**: Removed silent modification of user settings
2. **Duplicated Logic**: Centralized fallback detection
3. **Type Safety**: Fixed potential TypeError with Path objects
4. **Code Quality**: Prefixed unused variables, simplified boolean expressions
5. **Resource Leaks**: Proper session cleanup in all paths

---

## Previous Sessions Summary

### Session: November 18, 2025 (Morning) - Current
- Hybrid routing polish (PR #151)
- 1343 lines added, 29 new comprehensive tests
- Fixed 5 critical bugs from code review (cache type mismatch, inline imports, async I/O)
- Comprehensive security hardening (SSRF prevention)
- All 10 review issues addressed

### Session: November 16, 2025 (Evening)
- Implemented hybrid routing (PR #149)
- 989 lines of new code, 42 new tests
- Addressed 3 rounds of reviewer feedback
- Fixed critical type safety bug

### Session: November 16, 2025 (Afternoon)
- Centralized fallback logic (PR #148)
- SDK-friendly API design

### Session: November 16, 2025 (Early AM)
- PR #147: Critical fallback fixes
- PR #146: Documentation reduction

### Session: November 15, 2025
- Multi-provider LLM architecture (PR #144)
- Resource management patterns

### Session: November 14-15, 2025
- Pydantic migration (PRs #141, #142)
- Full schema validation
