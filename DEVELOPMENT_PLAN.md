# Development Plan for Mad Spark Alt

## Executive Summary

Mad Spark Alt is a hypothesis-driven analysis system implementing the QADI methodology with genetic evolution capabilities. The project has recently undergone major refactoring (PR #40) to implement true QADI methodology and has addressed critical bugs (PR #42). This plan outlines immediate fixes, enhancements, and future development tasks.

## Current State Analysis

### Recently Completed
- ✅ **True QADI Implementation** (PR #40): Replaced creative idea generation with hypothesis-driven consulting approach
- ✅ **Cost Tracking Fix** (PR #42): Fixed missing cost assignment in abduction phase
- ✅ **Evolution System Enhancements** (PR #38): Added caching, checkpointing, and LLM operators
- ✅ **Rich Terminal Rendering** (PR #33): Beautiful markdown output for CLI

### System Architecture
- **Core**: Simple QADI orchestrator with universal prompts
- **Evaluation**: Unified 5-criteria scoring (novelty, impact, cost, feasibility, risks)
- **Evolution**: AI-powered genetic algorithms with 50-70% API call reduction through caching
- **LLM Support**: Google (preferred), OpenAI, and Anthropic APIs

## Task Categories

### 1. IMMEDIATE FIXES (HIGH PRIORITY)

#### 1.1 Fix Method Signatures in benchmarks.py
**Priority**: 🔴 HIGH  
**Effort**: 2-5 minutes  
**Source**: PR #38 review feedback  
**Issue**: `_evolve_generation` method signature doesn't match parent class

```python
# Current (incorrect):
async def _evolve_generation(self, generation: int)

# Should be:
async def _evolve_generation(
    self,
    population: List[IndividualFitness],
    config: EvolutionConfig,
    context: Optional[str],
    generation: int,
) -> List[IndividualFitness]:
```

**Action Items**:
- [ ] Update method signature to match parent class
- [ ] Fix super() call to pass correct arguments
- [ ] Remove references to self._population (use parameter instead)
- [ ] Run tests to verify fix

### 2. CODE QUALITY IMPROVEMENTS (MEDIUM PRIORITY)

#### 2.1 Centralize Cost Estimation Logic
**Priority**: 🟡 MEDIUM  
**Effort**: 10-15 minutes  
**Source**: Code duplication in PR #38  
**Files**: `cost_estimator.py`, `llm_operators.py`

**Action Items**:
- [ ] Create central `calculate_llm_cost()` utility function
- [ ] Extract cost constants to single location
- [ ] Update all modules to use centralized logic
- [ ] Add unit tests for cost calculation

#### 2.2 Add Regression Tests for Cost Tracking
**Priority**: 🟡 MEDIUM  
**Effort**: 10-15 minutes  
**Source**: Bug fix in PR #42  
**Purpose**: Prevent regression of abduction phase cost tracking

**Action Items**:
- [ ] Create test case that verifies phase_results includes cost
- [ ] Test all QADI phases for cost inclusion
- [ ] Add assertion for total_llm_cost accuracy
- [ ] Document expected cost structure

### 3. TESTING & COVERAGE (MEDIUM PRIORITY)

#### 3.1 Improve Test Coverage
**Priority**: 🟡 MEDIUM  
**Effort**: 30-45 minutes  
**Current**: ~80% coverage (estimate)  
**Target**: 90%+ coverage

**Action Items**:
- [ ] Run coverage report to identify gaps
- [ ] Add tests for new QADI implementation
- [ ] Test edge cases in evolution system
- [ ] Add integration tests for CLI commands
- [ ] Test fallback behaviors for missing APIs

#### 3.2 Performance Benchmarking
**Priority**: 🟡 MEDIUM  
**Effort**: 20-30 minutes  
**Purpose**: Establish performance baselines

**Action Items**:
- [ ] Fix benchmarks.py method signatures first
- [ ] Create benchmark suite for QADI cycle
- [ ] Measure cache hit rates in practice
- [ ] Document performance metrics
- [ ] Create performance regression tests

### 4. DOCUMENTATION UPDATES (LOW PRIORITY)

#### 4.1 API Documentation
**Priority**: 🟢 LOW  
**Effort**: 20-30 minutes  
**Purpose**: Complete API reference

**Action Items**:
- [ ] Document SimpleQADIOrchestrator API
- [ ] Document UnifiedEvaluator API
- [ ] Update examples for new QADI approach
- [ ] Add migration guide from old system

#### 4.2 Architecture Diagrams
**Priority**: 🟢 LOW  
**Effort**: 15-20 minutes  
**Purpose**: Visual system overview

**Action Items**:
- [ ] Create QADI flow diagram
- [ ] Document evolution pipeline
- [ ] Show component interactions
- [ ] Add to DEVELOPMENT.md

### 5. FEATURE ENHANCEMENTS (FUTURE)

#### 5.1 Advanced Evolution Features
**Priority**: 🔵 FUTURE  
**Effort**: 2-4 hours  
**From**: Issue #36 remaining phases

**Potential Features**:
- [ ] Multi-objective optimization (Pareto frontiers)
- [ ] Dynamic strategy selection
- [ ] Real-time monitoring dashboard
- [ ] Advanced clustering algorithms
- [ ] Distributed evolution support

#### 5.2 Enhanced User Experience
**Priority**: 🔵 FUTURE  
**Effort**: 1-2 hours  

**Potential Features**:
- [ ] Interactive mode for QADI
- [ ] Progress bars for long operations
- [ ] Better error messages with recovery suggestions
- [ ] Export results to multiple formats
- [ ] Web UI for non-technical users

#### 5.3 Integration Capabilities
**Priority**: 🔵 FUTURE  
**Effort**: 2-3 hours  

**Potential Features**:
- [ ] REST API for QADI service
- [ ] Webhook support for async processing
- [ ] Database integration for result storage
- [ ] Integration with popular AI tools
- [ ] Plugin system for custom evaluators

## Implementation Order

### Session 1 (Next 30 minutes)
1. Fix benchmarks.py method signatures (5 min)
2. Centralize cost estimation logic (15 min)
3. Add regression tests for cost tracking (10 min)

### Session 2 (Following session)
1. Run and analyze coverage report
2. Add missing test cases
3. Create performance benchmarks

### Session 3 (Future)
1. Update API documentation
2. Create architecture diagrams
3. Plan next feature phase

## Success Metrics

- ✅ All tests passing (including new regression tests)
- ✅ No mypy errors
- ✅ Test coverage > 90%
- ✅ Benchmarks running successfully
- ✅ Cost calculation centralized (no duplication)
- ✅ Documentation complete and accurate

## Technical Debt to Address

1. **Deprecation Cleanup**: Remove deprecated modules after grace period
2. **Type Annotations**: Some complex types need better annotations
3. **Error Handling**: Standardize error messages and codes
4. **Logging**: Implement structured logging throughout
5. **Configuration**: Move hardcoded values to config

## Notes for Next Developer

- The system has moved from creative idea generation to hypothesis-driven analysis
- Google API is preferred due to reliability and cost
- Evolution system now has sophisticated caching - use it!
- Always run `mypy` before pushing (CI is strict)
- Template agents are deprecated - only use LLM agents
- The 5-criteria evaluation system is used consistently throughout

## References

- Original methodology: "Shin Logical Thinking" book
- Issue #36: Evolution enhancement roadmap
- PR #40: True QADI implementation details
- PR #42: Critical fixes and patterns
- CLAUDE.md: Project-specific patterns and conventions