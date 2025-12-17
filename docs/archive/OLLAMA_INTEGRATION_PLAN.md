# Ollama Integration Plan - Multi-Provider LLM Support

**Date:** 2025-01-15
**Status:** COMPLETE - MVP Shipped (see deviations below)
**Branch:** `feature/ollama-integration` → **Merged to main**

**Note:** This document describes the original implementation plan. See [../session_handover.md](../session_handover.md) for actual shipped implementation status.

## Executive Summary

This document captures the decision-making process and implementation plan for integrating Ollama (local LLM) support into Mad Spark Alt, creating a hybrid system that balances cost (Gemini API) and free local inference (Ollama gemma3:12b).

**Target Cost Reduction:** 70-90% via local Ollama for text-only queries

## Research Findings - Ollama Structured Output

**Source:** ollama.com/blog/structured-outputs, GitHub discussions (2024-2025)

**Key Capabilities:**
✅ Supports JSON schema via `format` parameter  
✅ Compatible with Pydantic: `format=MyModel.model_json_schema()`  
✅ Temperature=0 recommended for schema compliance  
⚠️ Some reliability issues reported with Gemma 3 12B  
✅ 3-layer fallback (Pydantic → JSON → text) will handle edge cases

## User Requirements (from discussion)

1. **Provider Selection:** Automatic (with `--provider` override)
2. **Default for text/images:** Always Ollama (maximize savings)
3. **Evolution operations:** Always Ollama
4. **Structured output:** Rely on Pydantic schemas
5. **Performance tolerance:** 2x slower than Gemini is acceptable

## Routing Logic

```
--provider gemini          → Use Gemini for everything
--provider ollama + docs   → ERROR (Ollama can't handle docs/URLs)
--provider ollama (text)   → Use Ollama for everything
--provider auto (default):
    IF --document OR --url → Gemini preprocessing → Ollama QADI
    ELSE                   → Ollama only (fallback to Gemini on failure)
```

## Definition of Done

**⚠️ Deviations from Original Plan:**
- **Phase 4 (Hybrid Orchestration):** SKIPPED for MVP - not required for basic functionality
- **Test Count:** 27 new tests (not 50+ as initially planned)
- **Existing Tests:** 968/971 passing (3 pre-existing failures unrelated to this PR)

**Code Complete (Actual Shipped MVP):**
- ✅ OllamaProvider with Pydantic schema support
- ✅ ProviderRouter with auto-selection + fallback
- ✅ Provider wiring into SimpleQADIOrchestrator (added post-review)
- ⏭️ Hybrid orchestration (deferred - can add in future PR)
- CLI `--provider` flag with validation

**Testing Complete:**
- All 962+ existing tests pass
- 50+ new Ollama tests (unit + integration)
- Real API tests for 5 user scenarios
- Performance benchmarks < 2x Gemini

**User Validation (Actual Status):**
1. ✅ Pure Ollama (text) - complete analysis, no errors
2. ✅ Pure Gemini (PDF+images) - multimodal processing works
3. ❌ Hybrid (PDF → Ollama) - **NOT IMPLEMENTED** (entire query goes to Gemini when documents/URLs present)
4. ✅ Evolution (Ollama) - semantic operators work
5. ⚠️ Fallback (Ollama down → Gemini) - CLI-only, not SDK/subcommands

**Quality Standards (Non-Negotiable):**
❌ No timeouts  
❌ No truncated outputs  
❌ No repeated content  
❌ No broken formatting  
❌ No cryptic errors  
✅ Clear provider status logging  
✅ Accurate cost tracking ($0.00 for Ollama)

**Documentation Complete:**
- README: Ollama setup guide
- ARCHITECTURE: Provider abstraction diagrams
- OLLAMA_SETUP.md: Installation instructions
- CLI help text: Provider selection examples
- CLAUDE.md: Learned patterns

## Implementation Phases (9-10 days)

### Phase 1: OllamaProvider Class (Day 1-2)
**Files:** `llm_provider.py`  
**Deliverables:**
- `OllamaProvider` implementing `LLMProviderInterface`
- Pydantic schema support via `format` parameter
- Image handling via base64
- Unit + integration tests

### Phase 2: Provider Router (Day 3)
**Files:** `provider_router.py` (new)  
**Deliverables:**
- Auto-selection logic
- Ollama → Gemini fallback
- Validation (block Ollama + documents)
- Tests for all routing scenarios

### Phase 3: CLI Integration (Day 4)
**Files:** `unified_cli.py`  
**Deliverables:**
- `--provider {auto|gemini|ollama}` flag
- Clear error messages
- Status display ("Using Ollama...")
- Help text updates

### Phase 4: Hybrid Orchestration (Day 5-6)
**Files:** `simple_qadi_orchestrator.py`  
**Deliverables:**
- Phase 0: Gemini document preprocessing
- Phases 1-4: Ollama QADI workflow
- Evolution: Ollama semantic operators
- Cost tracking (preprocessing + QADI)

### Phase 5: Comprehensive Testing (Day 7-8)
**Files:** `tests/test_ollama_*.py` (new)  
**Deliverables:**
- Unit tests (mocked Ollama API)
- Integration tests (real gemma3 model)
- User scenario tests (5 workflows)
- Performance benchmarks

### Phase 6: Documentation (Day 9-10)
**Files:** `README.md`, `ARCHITECTURE.md`, `docs/*`  
**Deliverables:**
- Updated setup instructions
- Provider selection guide
- Architecture diagrams
- Troubleshooting guide

## Cost Analysis

**Current (Pure Gemini):**
- QADI query: ~$0.02
- Evolution: ~$0.15
- Monthly (100 queries): $2-5

**Projected (Hybrid):**
- Pure Ollama (70%): $0.00
- Hybrid (20%): ~$0.01
- Pure Gemini (10%): ~$0.05
- Monthly: $0.50-1.00

**Savings: 70-90%**

## Success Metrics

- [ ] Zero timeouts in all test scenarios
- [ ] 100% Pydantic schema compliance
- [ ] Performance < 2x Gemini baseline
- [ ] 70%+ queries use Ollama
- [ ] New users complete setup in <15 min
- [ ] All documentation examples work as written

## Next Steps

1. Create `feature/ollama-integration` branch
2. Phase 1: OllamaProvider implementation (TDD)
3. Test as user after each commit
4. Report back when ALL phases complete

**Commit Protocol:**
- Logical checkpoints (after each test passes)
- Real API validation before commit
- User testing for every feature
