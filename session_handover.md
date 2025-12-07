# Session Handover

## Last Updated: December 7, 2025

---

## Current Work In Progress

**None** - Clean state, ready for new work.

---

## Recently Completed

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

## Session Learnings

### Structured Output Over Prompt Engineering (PR #161)
- **NEVER** request formatting in prompts when using structured output
- Use `Field(min_length=3, max_length=5)` in Pydantic schemas
- Prompts describe WHAT; schemas define HOW
- Anti-pattern: `"1. [item]"` + `List[str]` → double numbering
- Never use regex to fix formatting - fix prompt/schema instead

### CLI Flag Guarding Pattern (PR #158)
```python
mode_only_options = {'option': value if value != default else None}
if not mode_flag:
    used = [name for name, val in mode_only_options.items() if val]
    if used:
        show_error_with_hint()
```

### Cache Type Consistency (PR #151)
- Keep hash values as same type throughout cache lifecycle
- Float vs int comparison causes precision loss → silent invalidation

### Hybrid Routing Architecture (PR #149)
- Preprocess with Gemini → Run QADI with Ollama (free)
- Fail-fast when no valid content extracted

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
| Dec 7, 2025 | #161 | CLI output improvements, QADI phases |
| Nov 20, 2025 | #160 | Japanese UAT fixes |
| Nov 18-19 | #154, #157, #158 | Ollama paths, CLI syntax, evaluate flag |
| Nov 18 | #151 | Hybrid routing polish, 29 tests |
| Nov 16 | #148, #149 | Hybrid routing, centralized fallback |
| Nov 15-16 | #144, #146, #147 | Multi-provider, documentation |
| Nov 14-15 | #141, #142 | Pydantic migration |
