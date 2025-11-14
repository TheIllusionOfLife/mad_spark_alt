# Pydantic Structured Outputs Migration - Session Handover

## Branch: feature/pydantic-structured-outputs

## Status: Phase 3a Complete (Schema Generation) - Phase 3b Pending (Pydantic Validation)

## Commits Completed

### Commit 1: Phase 1 Foundation (90c0bbb)
test: add universal Pydantic schema models with multi-provider support

**Files Created**:
- `src/mad_spark_alt/core/schemas.py`: 9 Pydantic models (185 lines)
- `src/mad_spark_alt/core/schema_utils.py`: Conversion utilities (115 lines)
- `tests/test_schemas.py`: 29 comprehensive tests (631 lines)

**Key Features**:
- ✅ Standard JSON Schema output (lowercase "object", not "OBJECT")
- ✅ Score validation: `Field(ge=0.0, le=1.0)`
- ✅ Strict validation: `ConfigDict(extra="forbid")`
- ✅ Property ordering preserved
- ✅ Schema reusability via nested models

**Test Results**: 29/29 passing

---

### Commit 2: Phase 2 LLM Provider Integration (f845bab)
refactor: add multi-provider Pydantic schema support to LLM provider

**Files Modified**:
- `src/mad_spark_alt/core/llm_provider.py`: Accept Pydantic models
- `tests/test_llm_provider_pydantic.py`: 10 integration tests (302 lines)

**Key Changes**:
- ✅ `LLMRequest.response_schema: Union[Dict, type]`
- ✅ `LLMRequest.get_json_schema()` method
- ✅ `GoogleProvider` uses `get_json_schema()`
- ✅ Backward compatible with dict schemas

**Test Results**: 10/10 new tests + 930/930 existing tests passing

---

### Commit 3: Phase 3 QADI Phase Schema Generation (ab8e9c7 + fixes)
refactor: migrate QADI phase schemas to Pydantic models

**Scope**: Schema Generation Only (NOT full Pydantic validation)

**Files Modified**:
- `src/mad_spark_alt/core/phase_logic.py`: Updated `get_hypothesis_generation_schema()` and `get_deduction_schema()`
- `tests/test_phase_logic_pydantic.py`: 15 comprehensive validation tests (448 lines)
- `tests/test_structured_output_integration.py`: Updated for standard JSON Schema format

**What's Implemented**:
- ✅ `get_hypothesis_generation_schema()` returns `HypothesisListResponse.model_json_schema()`
- ✅ `get_deduction_schema()` returns `DeductionResponse.model_json_schema()`
- ✅ LLM API receives Pydantic-generated schemas for structured output
- ✅ Standard JSON Schema format (lowercase types)
- ✅ Schema validation tests passing

**What's NOT Implemented** (Future Work):
- ❌ Parsing still uses manual JSON with `.get()` and defaults
- ❌ NOT using `DeductionResponse.model_validate_json()` for automatic validation
- ❌ Missing benefits: type-safe access, clear error messages, automatic validation

**Test Results**: 15/15 new tests + 945/945 existing tests passing

**Impact**: Schemas are generated correctly and sent to LLM, but response parsing doesn't leverage Pydantic validation yet.

---

## Remaining Work

### Phase 3b: QADI Phase Pydantic Validation (CRITICAL GAP)

**Objective**: Replace manual JSON parsing with Pydantic validation in QADI phases

**Current Issue**: Phase 3 only implemented schema **generation**, not validation parsing

**Files to Modify**:
1. `src/mad_spark_alt/core/phase_logic.py` - `execute_deduction_phase()` (lines 565-614)
2. `src/mad_spark_alt/core/phase_logic.py` - `execute_abduction_phase()` (hypothesis parsing)

**Changes Needed**:
```python
# CURRENT (manual parsing):
data = json.loads(content)
for eval_data in data.get("evaluations", []):
    score_data = eval_data.get("scores", {})
    # ... manual extraction with .get() and defaults

# NEEDED (Pydantic validation):
try:
    result = DeductionResponse.model_validate_json(content)
    # Type-safe access: result.evaluations[0].scores.impact
except ValidationError as e:
    logger.warning(f"Pydantic validation failed: {e}")
    # Fall back to manual parsing
```

**Benefits**:
- ✅ Automatic validation (catches invalid LLM responses)
- ✅ Type-safe access (IDE autocomplete, no typos)
- ✅ Clear error messages (shows exactly what failed validation)
- ✅ Score range enforcement (0.0-1.0 validated automatically)

**Tests to Add**:
- Test Pydantic validation with invalid scores (> 1.0, < 0.0)
- Test graceful fallback when validation fails
- Test type-safe access patterns

**Estimated Time**: 1-2 hours

**Commit**: "refactor: use Pydantic validation in QADI phase parsing"

---

### Phase 4: Evolution Operator Schema Migration

**Objective**: Migrate evolution operators to use Pydantic schemas

**Files to Modify**:
1. `src/mad_spark_alt/evolution/semantic_utils.py` (Lines 183-221)
2. `src/mad_spark_alt/evolution/semantic_mutation.py` (Lines 343-349, 505-522, 680-784)
3. `src/mad_spark_alt/evolution/semantic_crossover.py` (Lines 575-594, 436-500)

**Changes**:
- Replace manual schema dicts with Pydantic models
- Update parsing: `MutationResponse.model_validate_json()`
- Update parsing: `BatchMutationResponse.model_validate_json()`
- Update parsing: `CrossoverResponse.model_validate_json()`
- Update parsing: `BatchCrossoverResponse.model_validate_json()`

**Tests to Update**:
- `tests/test_semantic_mutation.py`
- `tests/test_semantic_crossover.py`
- Add Pydantic validation tests

**Commit**: "refactor: migrate evolution operator schemas to Pydantic"

---

### Phase 5: Integration Testing

**Objective**: Verify Pydantic schemas work with real Gemini API

**File to Create**: `tests/test_multi_provider_schemas.py`

**Tests Required**:
1. Real Gemini API with `DeductionResponse` schema
2. Real Gemini API with `HypothesisListResponse` schema
3. Verify score validation (0.0-1.0 enforced by API)
4. Verify strict validation rejects extra fields
5. Verify property ordering consistency
6. Test error handling for invalid responses

**Command**:
```bash
GOOGLE_API_KEY=xxx pytest tests/test_multi_provider_schemas.py::test_real_api -v
```

**Commit**: "test: add Pydantic schema integration tests with real API"

---

### Phase 6: User Testing (CRITICAL - DO NOT SKIP!)

**Objective**: Verify system works end-to-end with real API

**Test Scenarios**:

1. **Basic QADI** (3+ different question types):
   ```bash
   msa "How can we reduce ocean plastic pollution?"
   msa "What's the best way to learn machine learning?"
   msa "How do I build a startup?"
   ```

2. **QADI + Evolution** (2+ configurations):
   ```bash
   msa "How can AI improve education?" --evolve --generations 3 --population 5
   msa "What's the future of renewable energy?" --evolve --generations 3 --population 8
   ```

3. **Multi-Perspective**:
   ```bash
   msa "How can we address climate change?" --perspectives environmental,technical,business
   ```

**Success Criteria** (ZERO TOLERANCE):
- ✅ NO timeouts
- ✅ NO truncated output
- ✅ NO repeated content
- ✅ NO format errors
- ✅ NO parsing failures
- ✅ NO validation errors

**If ANY issue**: Fix immediately, don't proceed

---

### Phase 7: Documentation

**Files to Create/Update**:

1. **NEW**: `docs/MULTI_PROVIDER_SCHEMAS.md`
   - Schema design philosophy
   - How to add new providers (OpenAI, Anthropic examples)
   - Schema conversion patterns
   - Troubleshooting guide

2. **UPDATE**: `ARCHITECTURE.md`
   - Add "Pydantic Schema Architecture" section
   - Document validation benefits
   - Show schema reusability patterns

3. **UPDATE**: `CLAUDE.md`
   - Add to "Project-Specific Patterns" section:
     - Pydantic Schema Best Practices
     - Validation constraint patterns
     - Multi-provider schema usage examples

**Commit**: "docs: document Pydantic schema architecture and multi-provider support"

---

## How to Resume

### 1. Checkout Branch
```bash
cd /path/to/your/mad_spark_alt
git checkout feature/pydantic-structured-outputs
```

### 2. Verify Current State
```bash
# All tests should pass
uv run pytest tests/ -m "not integration" -v
# Expected: 930+ passing

# Phase 1 tests
uv run pytest tests/test_schemas.py -v
# Expected: 29/29 passing

# Phase 2 tests
uv run pytest tests/test_llm_provider_pydantic.py -v
# Expected: 10/10 passing
```

### 3. Start Phase 4 (Evolution Operator Schema Migration)
```bash
# Review evolution operator schemas
cat src/mad_spark_alt/evolution/semantic_utils.py | grep -A 20 "get_mutation_schema"
cat src/mad_spark_alt/evolution/semantic_mutation.py | grep -A 20 "get_mutation_schema"

# Start writing tests (TDD)
# Add Pydantic validation tests for mutation and crossover operators

# Then migrate operators to use Pydantic schemas
```

---

## Design Decisions

### 1. Union Type for Backward Compatibility
**Decision**: `response_schema: Optional[Union[Dict[str, Any], type]]`

**Rationale**:
- Gradual migration (not breaking change)
- Existing dict schemas continue working
- New code can use Pydantic models

### 2. Explicit Conversion Method
**Decision**: `get_json_schema()` method vs automatic conversion

**Rationale**:
- Clear conversion point for debugging
- Allows future caching optimization
- Explicit > implicit

### 3. Standard JSON Schema
**Decision**: Output lowercase "object", "string", etc.

**Rationale**:
- Gemini API update accepts standard JSON Schema
- OpenAI, Anthropic use standard format
- Future-proof for new providers

### 4. Strict Validation Default
**Decision**: `ConfigDict(extra="forbid")` on all models

**Rationale**:
- Catch LLM hallucinations early
- Prevent schema drift
- Clear error messages

---

## Known Issues / Gotchas

### 1. Pydantic v2 Required
- Use `model_json_schema()`, not `schema()`
- Use `ConfigDict`, not `class Config`
- Use `model_validate_json()`, not `parse_raw()`

### 2. Type Checking with Union
```python
# CORRECT:
if isinstance(schema, type) and issubclass(schema, BaseModel):

# INCORRECT:
if isinstance(schema, BaseModel):  # Fails for model classes
```

### 3. Property Ordering
- Pydantic v2 preserves field order by default
- Important for Gemini 2.5+ consistency
- Order in model definition = order in JSON Schema

### 4. Validation Error Messages
```python
try:
    result = DeductionResponse.model_validate_json(json_str)
except ValidationError as e:
    # e.errors() provides detailed error info
    logger.error(f"Validation failed: {e.errors()}")
```

---

## Test Coverage

| Phase | Tests | Status |
|-------|-------|--------|
| Phase 1: Schema Models | 29 | ✅ Passing |
| Phase 2: LLM Provider | 10 | ✅ Passing |
| Phase 3: QADI Phases | 15 | ✅ Passing |
| Phase 4: Evolution Ops | TBD | ⏳ Pending |
| Phase 5: Integration | TBD | ⏳ Pending |
| **Existing Tests** | **945** | **✅ Passing** |

**Current Total**: 999 tests passing

---

## Estimated Time Remaining

- ✅ Phase 3a (Schema Generation): Complete
- Phase 3b (Pydantic Validation): 1-2 hours ⚠️ CRITICAL GAP
- Phase 4 (Evolution): 2-3 hours
- Phase 5 (Integration): 1 hour
- Phase 6 (User Testing): 2-3 hours ⚠️ CRITICAL
- Phase 7 (Docs): 1-2 hours
- Final checks: 1 hour

**Total**: 8-12 hours

---

## Success Checklist

Before declaring PR complete:

**Code & Tests**:
- [x] Phase 3a complete (QADI schema generation)
- [ ] Phase 3b complete (Pydantic validation parsing) ⚠️ CRITICAL GAP
- [ ] Phase 4 complete (Evolution operators)
- [ ] Phase 5 complete (Integration tests)
- [ ] All new tests passing
- [ ] All 930+ existing tests passing
- [ ] mypy type checking passes

**User Testing**:
- [ ] Basic QADI: 3+ questions tested successfully
- [ ] QADI + Evolution: 2+ configurations tested
- [ ] Multi-perspective QADI tested
- [ ] Zero timeouts in all scenarios
- [ ] Zero truncated outputs
- [ ] Zero repeated content
- [ ] Zero format errors
- [ ] Zero parsing failures

**Documentation**:
- [ ] MULTI_PROVIDER_SCHEMAS.md created
- [ ] ARCHITECTURE.md updated
- [ ] CLAUDE.md updated

**Final Steps**:
- [ ] CI tests pass locally
- [ ] Pull request created
- [ ] PR description includes all benefits and changes

---

## Quick Reference

### Pydantic Models Created

```python
# QADI Phase Schemas
from mad_spark_alt.core.schemas import (
    HypothesisScores,        # Score validation (0.0-1.0)
    Hypothesis,              # Single hypothesis
    HypothesisEvaluation,    # Hypothesis + scores
    DeductionResponse,       # Complete deduction output
    HypothesisListResponse,  # Abduction output
)

# Evolution Operator Schemas
from mad_spark_alt.core.schemas import (
    MutationResponse,        # Single mutation
    BatchMutationResponse,   # Batch mutations
    CrossoverResponse,       # Single crossover
    BatchCrossoverResponse,  # Batch crossover
)
```

### Usage Examples

```python
# 1. In LLM Request
from mad_spark_alt.core.schemas import DeductionResponse
from mad_spark_alt.core.llm_provider import LLMRequest

request = LLMRequest(
    user_prompt="Evaluate hypotheses...",
    response_schema=DeductionResponse,  # Pydantic model
    response_mime_type="application/json"
)

# 2. Parsing Response
response_text = llm_response.content
result = DeductionResponse.model_validate_json(response_text)

# 3. Accessing Validated Data
for eval in result.evaluations:
    print(f"H{eval.hypothesis_id}: {eval.scores.impact}")
```

---

## Context for Next Session

**What Works**:
- ✅ Pydantic models generate standard JSON Schema
- ✅ LLMRequest accepts both Pydantic models and dicts
- ✅ Automatic conversion via `get_json_schema()`
- ✅ All existing tests still pass (backward compatible)
- ✅ Score validation constraints in place (0.0-1.0)
- ✅ Strict validation enabled (extra fields rejected)

**What's Next**:
- Migrate QADI phase schemas (phase_logic.py)
- Migrate evolution operator schemas
- Integration testing with real API
- Comprehensive user testing
- Documentation

**Why This Matters**:
- Gemini API now supports standard JSON Schema (announcement)
- Enables multi-provider compatibility (OpenAI, Anthropic, local LLMs)
- Automatic validation prevents invalid LLM outputs
- Type-safe code with IDE support
- Future-proof architecture

---

Last Updated: 2025-11-14 (Session after Phase 3 completion)
