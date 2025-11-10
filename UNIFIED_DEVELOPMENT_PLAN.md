# 🎯 Unified Development Plan - Mad Spark Alt
**Date**: November 10, 2025
**Status**: Active
**Last Updated**: 2025-11-10

**Synthesized from**:
- Project status analysis (session_handover.md, README.md)
- Architectural review (unified_refactoring.txt)
- UX review (unified_ux.txt)

---

## Executive Summary

After analyzing three comprehensive perspectives (project status, architectural review, UX review), I've identified **one critical architectural risk** that must be addressed first, followed by a balanced mix of technical debt reduction and high-impact user experience improvements.

### 🚨 Critical Issue Discovered
**CLI/SDK Divergence**: The CLI contains a custom `SimplerQADIOrchestrator` with divergent prompt logic separate from the core library's `SimpleQADIOrchestrator`. This creates:
- Dual maintenance burden
- Inconsistent behavior between CLI and SDK
- Testing complexity
- Risk of feature drift

**This was not in session_handover.md but is flagged as critical in the refactoring review.**

---

## 📊 Prioritization Framework

Using a **Risk × Impact × Effort** matrix:

| Priority | Criteria |
|----------|----------|
| **P0 (Critical)** | High risk, blocks future work, or breaks user trust |
| **P1 (High)** | High impact, enables other features, clear user value |
| **P2 (Medium)** | Important but not urgent, incremental improvements |
| **P3 (Future)** | Nice-to-have, exploratory, requires research |

---

## 🎯 Priority 0: Critical Architectural Fixes (Week 1: 3-5 days)

### **Why This First?**
The CLI/SDK divergence is a **systemic risk** that will compound over time. Every feature added to either CLI or SDK must now be duplicated. This blocks clean implementation of future features.

### Task 0.1: Restore CLI/SDK Parity ✅ COMPLETE
**Source**: unified_refactoring.txt Priority 2, Task 3
**Critical Risk**: CLI has custom orchestrator with divergent logic
**Status**: ✅ **RESOLVED** on 2025-11-10
**Actual Time**: ~6 hours (investigation: 4h, implementation: 1h, testing: 1h)

**Implementation Completed**:
1. ✅ **Phase 1: Investigation** - Created comprehensive analysis document (CLI_SDK_DIVERGENCE_ANALYSIS.md)
2. ✅ **Phase 2: Code Changes** - Removed SimplerQADI* classes (-23 lines), updated instantiation
3. ✅ **Phase 3: Test Updates** - Updated 6 mock locations, added parity tests
4. ✅ **Phase 4: Verification** - All 849 tests pass
5. ✅ **Phase 5: Real API Testing**:
   - Basic QADI: 50.2s, $0.0064 ✅
   - Evolution: 116.7s, $0.0063 ✅
   - JSON/MD export: Both work ✅
   - Multimodal: Images processed ✅
   - No timeouts, truncation, or errors ✅

**Success Criteria - ALL MET**:
- ✅ Single QADI implementation used by both CLI and SDK
- ✅ All 849 tests pass (was 844, added 5 parity tests)
- ✅ CLI quality verified with real API (no degradation)
- ✅ No code duplication between CLI and SDK
- ✅ Phase 1 now uses detailed SDK prompts (improved quality)

**Result**: CLI/SDK parity achieved, -23 lines, unified behavior

---

### Task 0.2: Fix "Execute on Import" Warnings ✅ COMPLETE
**Source**: unified_refactoring.txt Priority 1, Task 2
**Risk**: Deprecation warnings execute on import, cluttering user output
**Status**: ✅ **COMPLETED** on 2025-11-10
**Actual Time**: ~2 hours
**PR**: [#134](https://github.com/TheIllusionOfLife/mad_spark_alt/pull/134)

**Implementation Completed**:
1. ✅ Removed module-level warnings from 3 deprecated modules:
   - `smart_orchestrator.py`
   - `answer_extractor.py`
   - `robust_json_handler.py`
2. ✅ Implemented `__getattr__` lazy import mechanism in `core/__init__.py`
   - Defers warning until actual use
   - Caches imports to prevent duplicate warnings
   - Maintains backward compatibility
3. ✅ Added comprehensive test suite (`test_deprecation_warnings.py`)
   - 11 tests covering all warning scenarios
   - Tests for backward compatibility
   - All tests passing
4. ✅ Real API validation successful (CLI works with no warnings)

**Success Criteria - ALL MET**:
- ✅ No warnings on `import mad_spark_alt`
- ✅ No warnings on `import mad_spark_alt.core`
- ✅ Warnings only when deprecated code is explicitly imported
- ✅ All tests pass (11 new tests + all existing)
- ✅ Backward compatibility maintained

**Result**: Clean imports for users, warnings only on explicit deprecated module usage

**Branch**: `fix/deprecation-warnings`

---

## 🚀 Priority 1: High-Impact Quick Wins (Week 1: 2-3 days)

### Task 1.1: Live Progress Indicators ⏱️ 6-8 hours
**Source**: unified_ux.txt Priority 0, Task 3
**Impact**: **CRITICAL** - Eliminates "is it frozen?" anxiety during evolution

**Why This is P1**: Mentioned in **all UX reviews** as critical pain point. Low effort, huge user satisfaction impact.

**Implementation**:
```python
from rich.live import Live
from rich.table import Table

# In genetic_algorithm.py
with Live(self._create_progress_table(), refresh_per_second=4) as live:
    for generation in range(self.config.num_generations):
        live.update(self._create_progress_table(
            current_gen=generation,
            best_fitness=max(fitnesses),
            avg_fitness=mean(fitnesses),
            phase="Mutation" or "Evaluation" or "Selection"
        ))
```

**Display**:
```
┌─ Evolution Progress ─────────────────────────┐
│ Phase: Evaluation                             │
│ Generation: 3/5 (60%)                        │
│ Best Fitness: 0.87 ↑ 0.12                    │
│ Avg Fitness: 0.74                            │
│ ETA: 45 seconds                              │
└───────────────────────────────────────────────┘
```

**Success Criteria**:
- ✅ Real-time updates during QADI phases
- ✅ Evolution shows generation, fitness, ETA
- ✅ No performance degradation
- ✅ Works with `nohup` (graceful fallback to simple logging)

**Branch**: `feat/progress-indicators`

---

### Task 1.2: Remove Dead Dependencies ✅ COMPLETE
**Source**: unified_refactoring.txt Priority 1, Task 1
**Impact**: Reduces maintenance burden, speeds up installs
**Status**: ✅ **COMPLETED** on 2025-11-10
**Actual Time**: ~30 minutes
**PR**: [#137](https://github.com/TheIllusionOfLife/mad_spark_alt/pull/137)

**Implementation Completed**:
1. ✅ Removed `openai>=1.0.0` and `anthropic>=0.3.0` from pyproject.toml
2. ✅ Updated lock file with `uv lock`
3. ✅ Updated `llm_provider.py` docstring (Google-only)
4. ✅ Updated 4 example files (qadi_demo, llm_questioning_demo, llm_abductive_demo, llm_deductive_demo)
5. ✅ Updated documentation (cli_usage.md, docs/README.md)
6. ✅ Verified no openai/anthropic imports remain in src/ or tests/
7. ✅ All 861 tests pass
8. ✅ CLI tested with real API - works correctly (55.4s, $0.0070)
9. ✅ CI checks pass (test: 5m7s, build: 12s, CodeRabbit: pass)

**Success Criteria - ALL MET**:
- ✅ `pyproject.toml` only lists Google Gemini as LLM provider (15 dependencies, down from 17)
- ✅ No import errors - verified with grep
- ✅ `uv sync` completes faster (~0.025s)
- ✅ All tests pass (861 passed, 1 skipped)
- ✅ ~100MB disk space saved
- ✅ Documentation accurately reflects Google-only architecture

**Result**: Cleaner dependencies, faster installs, no confusion about supported providers

**Branch**: `chore/remove-dead-dependencies`

---

### Task 1.3: Centralize Magic Numbers ⏱️ 3-4 hours
**Source**: unified_refactoring.txt Priority 1, Task 3
**Impact**: Improves code maintainability

**Files to Audit**:
- `unified_cli.py` (timeouts, defaults)
- `semantic_mutation.py` (temperature, token limits)
- `genetic_algorithm.py` (population constraints)

**Create**:
```python
# src/mad_spark_alt/core/config.py
@dataclass(frozen=True)
class SystemConstants:
    # Evolution
    MIN_POPULATION: int = 2
    MAX_POPULATION: int = 10
    MIN_GENERATIONS: int = 2
    MAX_GENERATIONS: int = 5

    # Timeouts
    PHASE_TIMEOUT_BASE: int = 90
    PHASE_TIMEOUT_MULTIPLIER: int = 5
    MAX_TOTAL_TIMEOUT: int = 900

    # LLM
    DEFAULT_TEMPERATURE: float = 0.8
    BREAKTHROUGH_TEMPERATURE: float = 0.95
    DEFAULT_MAX_TOKENS: int = 1500

CONSTANTS = SystemConstants()
```

**Success Criteria**:
- ✅ No hardcoded numbers in business logic
- ✅ Single source of truth for system limits
- ✅ Easy to adjust for experimentation
- ✅ All tests pass with constants

**Branch**: `refactor/centralize-constants`

---

## 🔧 Priority 2: Performance & Core Improvements (Week 2-3: 8-12 days)

### Task 2.1: Optimize Diversity Calculation (O(n²) → O(n log n)) ⏱️ 8-10 hours
**Source**: Both original plan AND unified_refactoring.txt Priority 5
**Impact**: **Product-enabling** - unlocks populations of 20+

**Current Problem**:
```python
# diversity_calculator.py - O(n²)
for i in range(len(population)):
    for j in range(i+1, len(population)):
        similarity = jaccard_similarity(ideas[i], ideas[j])
```

**Solution**: Implement Approximate Nearest Neighbors
```python
from sklearn.neighbors import NearestNeighbors
# or: import faiss

# 1. Convert ideas to embeddings (if semantic)
# 2. Build index
# 3. Query k-nearest neighbors only
# Time: O(n log n) for index build + O(log n) per query
```

**Implementation Phases**:
1. **Benchmark current** (2 hours)
   - Test populations: 5, 10, 15, 20, 25, 30
   - Measure time and memory

2. **Implement ANN** (4 hours)
   - Use Faiss or sklearn NearestNeighbors
   - Maintain same diversity metric output

3. **Validate** (2 hours)
   - Verify diversity scores are similar
   - Test with populations up to 50
   - Real API evolution test

**Success Criteria**:
- ✅ Handles population size 20+ without degradation
- ✅ <5% difference in diversity scores vs O(n²)
- ✅ 10x+ speedup for population=20
- ✅ All evolution tests pass

**Branch**: `feat/diversity-optimization`

---

### Task 2.2: Deconstruct CLI Monolith (1,500 → ~400 lines) ⏱️ 12-16 hours
**Source**: unified_refactoring.txt Priority 2, Task 1
**Impact**: Maintainability, testability

**Current Structure**:
```
unified_cli.py (1,500 lines)
├── Click command definitions
├── Rich rendering logic
├── SimplerQADIOrchestrator (duplicate!)
├── Evolution handling
├── Multimodal processing
└── Output formatting
```

**Target Structure**:
```
src/mad_spark_alt/cli/
├── __init__.py
├── main.py (150 lines)          # Click wiring only
├── commands/
│   ├── qadi.py (200 lines)      # QADI command handler
│   ├── evolve.py (150 lines)    # Evolution command handler
│   └── evaluate.py (100 lines)  # Evaluation commands
├── display/
│   ├── formatters.py (200 lines) # Output formatting
│   ├── tables.py (150 lines)     # Rich table builders
│   └── progress.py (100 lines)   # Progress indicators
└── utils/
    └── validators.py (100 lines) # Input validation
```

**Implementation**:
1. **Create structure** (2 hours)
2. **Extract display logic** (4 hours)
3. **Extract command handlers** (6 hours)
4. **Update imports and tests** (3 hours)
5. **Remove SimplerQADIOrchestrator, use SimpleQADIOrchestrator** (3 hours)

**Success Criteria**:
- ✅ No file >300 lines
- ✅ Clear separation of concerns
- ✅ All CLI tests pass
- ✅ CLI uses core library orchestrator

**Branch**: `refactor/cli-structure`

---

### Task 2.3: Consolidate Text Cleaning Logic ⏱️ 4 hours
**Source**: unified_refactoring.txt Priority 2, Task 2
**Impact**: DRY principle, single source of truth

**Current Duplication** (3 locations):
1. `unified_cli.py` - ANSI cleaning for display
2. `json_utils.py` - JSON extraction cleaning
3. `parsing_utils.py` - Hypothesis text cleaning

**Create**:
```python
# src/mad_spark_alt/utils/text_cleaning.py
class TextCleaner:
    @staticmethod
    def remove_ansi(text: str) -> str:
        """Remove ANSI escape codes"""

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace and line breaks"""

    @staticmethod
    def extract_json_content(text: str) -> str:
        """Extract JSON from markdown/mixed content"""

    @staticmethod
    def clean_for_llm(text: str) -> str:
        """Full cleaning pipeline for LLM input"""
```

**Success Criteria**:
- ✅ Single implementation used by all 3 modules
- ✅ All parsing tests pass
- ✅ No behavioral changes
- ✅ Comprehensive unit tests for edge cases

**Branch**: `refactor/unify-text-cleaning`

---

### Task 2.4: Fix Silent Validation Failures ⏱️ 2 hours
**Source**: unified_refactoring.txt Priority 5, Task 2
**Impact**: Better error messages, prevents silent bugs

**Current Problem**:
```python
# EvolutionConfig.validate() returns False silently
if not config.validate():
    # What failed? User doesn't know!
    return None
```

**Fix**:
```python
def validate(self) -> None:
    """Validate configuration. Raises ValueError with clear message."""
    if self.population_size < 2:
        raise ValueError(
            f"Population size must be ≥2, got {self.population_size}. "
            f"Tip: Use --population 5 for standard runs."
        )

    if self.max_parallel_evaluations > self.population_size:
        raise ValueError(
            f"max_parallel_evaluations ({self.max_parallel_evaluations}) "
            f"cannot exceed population_size ({self.population_size})"
        )
```

**Success Criteria**:
- ✅ All validation errors raise ValueError
- ✅ Error messages include current value + expected range + helpful tip
- ✅ Update all callers to handle ValueError
- ✅ Add tests for each validation case

**Branch**: `fix/validation-errors`

---

## 🎨 Priority 3: UX Enhancements (Week 3-4: 6-8 days)

### Task 3.1: First-Run Wizard ⏱️ 4-6 hours
**Source**: unified_ux.txt Priority 0, Task 1
**Impact**: Smooth onboarding experience

**Implementation**:
```bash
$ msa setup
┌─ Mad Spark Alt Setup Wizard ───────────────────┐
│ Let's get you set up! (2 minutes)              │
└─────────────────────────────────────────────────┘

Step 1/3: API Key
Where is your Google API key?
  [1] I have one (enter manually)
  [2] I need to create one (open guide)
> 1

Enter your Gemini API key: **********************

Testing connection... ✓ Success!

Step 2/3: Default Settings
What's your primary use case?
  [1] Quick analysis (faster, cheaper)
  [2] Deep research (thorough, higher cost)
  [3] Creative ideation (exploratory)
> 1

Step 3/3: Confirmation
Created .env with:
  - GOOGLE_API_KEY: ••••••••1234
  - Default preset: quick-analysis

Ready to go! Try: msa "How can we reduce waste?"
```

**Success Criteria**:
- ✅ Creates `.env` file with API key
- ✅ Tests API connection
- ✅ Offers to create key if missing
- ✅ Sets sensible defaults based on use case
- ✅ Skip if `.env` already exists (or offer to reconfigure)

**Branch**: `feat/setup-wizard`

---

### Task 3.2: Demo Mode (Zero-Setup Experience) ⏱️ 3-4 hours
**Source**: unified_ux.txt Priority 0, Task 2
**Impact**: Immediate value demonstration

**Implementation**:
```bash
$ msa demo
┌─ Mad Spark Alt Demo ────────────────────────────┐
│ Here's what Mad Spark can do for you           │
│ (Running cached example - no API key needed)   │
└──────────────────────────────────────────────────┘

Question: How can we make cities more sustainable?

[Shows pre-cached, beautifully formatted output]

──────────────────────────────────────────────────
Ready to try with your question?
1. Run 'msa setup' to configure your API key
2. Then try: msa "Your question here"
```

**Implementation**:
- Embed a pre-computed, high-quality QADI result
- Show full output with Rich formatting
- Include evolution example too
- Point to setup wizard

**Success Criteria**:
- ✅ Works without API key
- ✅ Shows representative output
- ✅ Clear call-to-action to setup
- ✅ <1 second to display

**Branch**: `feat/demo-mode`

---

### Task 3.3: Preset System ⏱️ 6-8 hours
**Source**: unified_ux.txt Priority 1, Task 3
**Impact**: Reduces cognitive load for users

**Implementation**:
```python
# src/mad_spark_alt/cli/presets.py
PRESETS = {
    "quick": {
        "temperature": 0.7,
        "evolve": False,
        "num_hypotheses": 3,
        "description": "Fast analysis for quick insights"
    },
    "balanced": {
        "temperature": 0.8,
        "evolve": True,
        "generations": 2,
        "population": 5,
        "description": "Balanced quality and speed"
    },
    "creative": {
        "temperature": 1.2,
        "evolve": True,
        "generations": 3,
        "population": 8,
        "use_semantic_operators": True,
        "description": "Deep exploration with novel ideas"
    },
    "research": {
        "temperature": 0.6,
        "evolve": True,
        "generations": 4,
        "population": 10,
        "diversity_method": "semantic",
        "description": "Thorough analysis with high diversity"
    }
}
```

**CLI Usage**:
```bash
# List presets
msa --list-presets

# Use preset
msa "Question" --preset creative

# Preset + override
msa "Question" --preset balanced --temperature 1.0
```

**Success Criteria**:
- ✅ 4-5 well-designed presets
- ✅ `--list-presets` shows descriptions
- ✅ Flags can override preset values
- ✅ Help text explains when to use each

**Branch**: `feat/presets`

---

### Task 3.4: Cost Estimate & Budget Guard ⏱️ 4-6 hours
**Source**: unified_ux.txt Priority 2, Task 1
**Impact**: User trust, prevents surprise bills

**Implementation**:
```bash
$ msa "Question" --evolve --generations 5 --population 10 --cost-estimate
┌─ Cost Estimate ─────────────────────────────────┐
│ QADI Analysis:                   $0.012         │
│ Evolution (5 gen × 10 pop):      $0.050         │
│ Fitness Evaluation:              $0.030         │
│ Diversity (semantic):            $0.001         │
│                                                  │
│ Total Estimated Cost:            $0.093         │
│ Estimated Time:                  ~6 minutes     │
└──────────────────────────────────────────────────┘

Proceed? [y/N]

# Budget guard
$ msa "Question" --evolve --budget 0.02
⚠️  Warning: Estimated cost ($0.09) exceeds budget ($0.02)
Auto-adjusting: Using population=3, generations=2
New estimate: $0.018 ✓
```

**Success Criteria**:
- ✅ Accurate cost estimation (within 20%)
- ✅ `--cost-estimate` shows breakdown without running
- ✅ `--budget` auto-adjusts or rejects
- ✅ Actual cost logged after completion

**Branch**: `feat/cost-controls`

---

## 🔮 Priority 4: Advanced Features (Week 5+: Future)

### Task 4.1: Directed Evolution Mode ⏱️ 12-16 hours
**Source**: Original plan + archived development plan
**Impact**: Fundamentally improves evolution quality

**Deferred because**: Requires diversity optimization (Task 2.1) to be complete first

**Implementation** (when ready):
- Multi-stage strategy: Explore (gen 1-2) → Exploit (gen 3-4) → Synthesize (gen 5)
- Gradient-based operator selection
- Elite-specific enhancement mutations

---

### Task 4.2: Interactive REPL Mode ⏱️ 16-20 hours
**Source**: unified_ux.txt Priority 3, Task 1
**Impact**: Transforms tool into thinking partner

**Example**:
```bash
$ msa interactive
msa> analyze "How can we reduce traffic?"

[Runs QADI, shows results]

msa> refine hypothesis 2
[Focuses evolution on hypothesis 2]

msa> compare hypothesis 1 vs 3
[Shows side-by-side comparison]

msa> export markdown results.md
✓ Saved to results.md

msa> exit
```

---

### Task 4.3: Session Management ⏱️ 8-10 hours
**Source**: unified_ux.txt Priority 3, Task 2
**Impact**: Enables iteration on previous work

```bash
msa list                    # Show past sessions
msa show <id>               # View details
msa --load-session <id>     # Continue from previous
```

---

## 📅 Recommended Execution Timeline

### **Week 1: Critical Fixes + Quick Wins** (5 days)
**Goal**: Address architectural risk, improve UX significantly

**Day 1-2**:
- ✅ Task 0.1: CLI/SDK Parity (CRITICAL)
- ✅ Task 0.2: Fix Deprecation Warnings
- ✅ Task 1.2: Remove Dead Dependencies

**Day 3**:
- ✅ Task 1.3: Centralize Constants

**Day 4-5**:
- ✅ Task 1.1: Live Progress Indicators (HIGH IMPACT UX)

**Deliverables**:
- Single QADI implementation
- Clean imports (no warnings)
- Real-time progress during evolution

---

### **Week 2-3: Performance & Refactoring** (8 days)

**Day 6-7**:
- ✅ Task 2.1: Diversity Optimization (enables larger populations)

**Day 8-9**:
- ✅ Task 2.3: Consolidate Text Cleaning
- ✅ Task 2.4: Fix Validation Errors

**Day 10-13**:
- ✅ Task 2.2: Deconstruct CLI Monolith

**Deliverables**:
- Support for population sizes 20+
- Clean, modular CLI architecture
- Better error messages

---

### **Week 3-4: UX Polish** (6 days)

**Day 14-15**:
- ✅ Task 3.1: First-Run Wizard
- ✅ Task 3.2: Demo Mode

**Day 16-17**:
- ✅ Task 3.3: Preset System

**Day 18-19**:
- ✅ Task 3.4: Cost Controls

**Deliverables**:
- Smooth onboarding
- Zero-setup demo
- User-friendly presets
- Cost transparency

---

### **Week 5+: Advanced Features** (Future)
- Directed Evolution Mode
- Interactive REPL
- Session Management
- Visualization tools

---

## 🎯 Current Status (As of 2025-11-10)

### **Completed**:
- ✅ All refactoring (14/14 items, 100% complete)
- ✅ CLI consolidation (PR #126)
- ✅ Result export system (PR #130)
- ✅ Multimodal support (PR #122, #124, #125)
- ✅ Task 0.1: CLI/SDK Parity (PR #131)
- ✅ Task 0.2: Fix Execute on Import Warnings (PR #134)
- ✅ Task 1.2: Remove Dead Dependencies (PR #137)
- ✅ 861/861 tests passing, 1 skipped

### **In Progress**:
- None currently

### **Blocked/Deferred**:
- None currently

---

## 📊 Summary Comparison

| Plan Source | Focus | Top Priority |
|-------------|-------|--------------|
| **Original Analysis** | Project status + obvious next steps | Diversity optimization, docs cleanup |
| **Refactoring Review** | Technical debt + architecture | CLI/SDK divergence, dead code |
| **UX Review** | User experience + onboarding | Progress indicators, first-run wizard |
| **Unified Plan** | Balance all three | CLI/SDK parity → Progress indicators → Performance |

---

## 📝 Key Principles

Following CLAUDE.md guidelines:
- ✅ **TDD First**: Write tests before implementation
- ✅ **Feature Branch**: Create branch before any work
- ✅ **Real API Validation**: Test with actual Gemini API
- ✅ **Systematic Approaches**: No shortcuts
- ✅ **Cost Tracking**: Monitor LLM costs
- ✅ **Documentation**: Update as you go

---

## 🔗 Related Documents

- **Session Handover**: `session_handover.md` - Development history and learnings
- **Architecture**: `ARCHITECTURE.md` - System design
- **README**: `README.md` - User-facing documentation
- **Archived Plans**: `docs/archive/` - Completed planning documents

---

**Last Updated**: 2025-11-10
**Next Review**: After Week 1 completion
