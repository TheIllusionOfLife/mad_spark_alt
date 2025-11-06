# Implementation Plan: Performance & Evolution Enhancements

## Overview
This plan addresses two priority tasks identified in the README:
1. Performance Optimization: Diversity Calculation (O(n²) complexity issue)
2. Batch Semantic Operators Enhancement (breakthrough mutations for high-scoring ideas)

## Phase 1: Semantic Diversity with Gemini Embeddings

### Objective
Replace O(n²) word-based diversity calculation with efficient semantic diversity using Gemini embeddings API.

### Implementation Steps

1. **Create GeminiDiversityCalculator** (`src/mad_spark_alt/evolution/gemini_diversity.py`):
   ```python
   class GeminiDiversityCalculator:
       - Uses Gemini text-embedding-004 model
       - Task type: SEMANTIC_SIMILARITY
       - Dimension: 768 (recommended for semantic similarity)
       - Handles 2048 token limit per text
       - Returns diversity score 0-1
   ```

2. **Add Gemini embedding support** to `LLMProvider`:
   - Add `get_embedding()` method
   - Handle embedding API requests
   - Track embedding costs separately

3. **Create DiversityCalculator interface**:
   - Abstract base class for diversity calculators
   - Implementations: JaccardDiversityCalculator (current), GeminiDiversityCalculator
   - Configuration option to choose calculator

4. **Update FitnessEvaluator**:
   - Accept DiversityCalculator in constructor
   - Default to GeminiDiversityCalculator for semantic understanding
   - Fallback to JaccardDiversityCalculator if API fails

5. **Testing & Benchmarking**:
   - Compare diversity scores between methods
   - Measure performance improvement
   - Validate semantic diversity is more meaningful

### Benefits
- O(n) complexity instead of O(n²)
- True semantic understanding of idea diversity
- Better evolution outcomes through meaningful diversity maintenance

## Phase 2: Name Unification (fitness_score → overall_fitness)

### Objective
Standardize on `overall_fitness` throughout codebase to eliminate confusion.

### Implementation Steps

1. **Update semantic_operators.py**:
   - Replace all `fitness_score` metadata checks with `overall_fitness`
   - Update `_is_high_scoring_idea()` method

2. **Update test files**:
   - Search and replace in all test files
   - Ensure mock data uses correct field name

3. **Add migration logic**:
   - In checkpoint loading, rename `fitness_score` to `overall_fitness`
   - Maintain backward compatibility

4. **Update documentation**:
   - Clarify that only `overall_fitness` exists
   - Update any references in comments

## Phase 3: Breakthrough Mutation Validation

### Objective
Empirically validate whether breakthrough mutations produce better results before implementing batch support.

### Implementation Steps

1. **Create A/B testing framework** (`tests/evolution/test_breakthrough_effectiveness.py`):
   ```python
   class BreakthroughEffectivenessTest:
       - Test different thresholds: [0.7, 0.75, 0.8, 0.85]
       - Test different temperatures: [0.85, 0.9, 0.95, 1.0]
       - Measure improvement, success rate, cost efficiency
       - Statistical significance testing
   ```

2. **Implement single mutation support first**:
   - Complete the breakthrough mutation implementation in `mutate_single()`
   - Ensure proper prompt usage and temperature settings
   - Track mutation type in results

3. **Run controlled experiments**:
   - 20+ high-scoring ideas per configuration
   - Compare regular vs breakthrough mutations
   - Measure fitness improvement distribution
   - Calculate cost per successful improvement

4. **Decision criteria**:
   - Breakthrough mutations must show >15% better improvement
   - Success rate must be comparable or better
   - Cost increase must be justified by improvement

## Phase 4: Batch Breakthrough Implementation (Conditional)

### Objective
Only implement if Phase 3 proves breakthrough mutations are valuable.

### Implementation Steps

1. **Separate batch processing**:
   - Split ideas by fitness threshold
   - Process breakthrough and regular batches separately
   - Maintain original order in results

2. **Optimize prompts**:
   - Create `BREAKTHROUGH_BATCH_PROMPT`
   - Include mutation type assignments
   - Ensure 200+ word requirements

3. **Cache enhancement**:
   - Store mutation types in cache
   - Track breakthrough vs regular performance

4. **Performance testing**:
   - Ensure batch efficiency is maintained
   - Monitor token usage and costs

## Implementation Order

1. **Week 1**: Phase 2 (Name Unification) - Quick win, improves code clarity
2. **Week 1-2**: Phase 1 (Semantic Diversity) - High impact on all evolution runs
3. **Week 2**: Phase 3 (Breakthrough Validation) - Empirical testing
4. **Week 3**: Phase 4 (Batch Implementation) - Only if Phase 3 proves value

## Success Metrics

1. **Diversity Calculation**:
   - Performance: 10x+ speedup for populations >50
   - Quality: More meaningful diversity scores
   - Evolution: Better idea variety in final populations

2. **Breakthrough Mutations**:
   - Improvement: >15% better than regular mutations
   - Cost efficiency: <2x cost for >1.5x improvement
   - Success rate: ≥50% of mutations improve fitness

## Risk Mitigation

1. **API Costs**: Monitor Gemini embedding costs, implement caching
2. **Backward Compatibility**: Maintain fallback to word-based diversity
3. **Breakthrough Complexity**: Start with single mutations, scale only if proven
4. **Testing Coverage**: Comprehensive tests before production deployment

## Technical Details

### Gemini Embeddings Configuration
```python
embedding_config = {
    "model": "models/text-embedding-004",
    "task_type": "SEMANTIC_SIMILARITY",
    "output_dimensionality": 768,
    "title": None,  # Optional, for better quality
}
```

### Diversity Calculation Comparison

**Current (Jaccard)**:
- Time: O(n²) 
- 100 ideas = 4,950 comparisons
- Measures: Word overlap
- Quality: Surface-level similarity

**Proposed (Gemini)**:
- Time: O(n) API calls + O(n²) cosine similarity (fast)
- 100 ideas = 100 API calls + fast vector math
- Measures: Semantic meaning
- Quality: Deep conceptual similarity

### Cost Analysis

**Gemini Embeddings**:
- ~500 tokens per idea (average)
- 100 ideas = 50,000 tokens per generation
- Cost: Check current Gemini pricing
- Caching reduces repeated evaluations

**Breakthrough Mutations**:
- 2x tokens (regular: 1000, breakthrough: 2000)
- Higher temperature may increase token usage
- Only applies to top 20% of population

## Code Examples

### Semantic Diversity Implementation
```python
class GeminiDiversityCalculator(DiversityCalculator):
    async def calculate_diversity(self, population: List[IndividualFitness]) -> float:
        if len(population) < 2:
            return 1.0
            
        # Get embeddings with caching
        embeddings = await self._get_embeddings(population)
        
        # Calculate pairwise similarities efficiently
        similarity_matrix = cosine_similarity(embeddings)
        
        # Extract upper triangle (excluding diagonal)
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        avg_similarity = np.mean(upper_triangle)
        
        return 1.0 - avg_similarity
```

### Breakthrough Testing Framework
```python
async def compare_mutations(idea: GeneratedIdea, fitness: float) -> Dict:
    # Regular mutation
    regular = await operator.mutate_single(
        idea, 
        temperature=0.8,
        use_breakthrough=False
    )
    
    # Breakthrough mutation
    breakthrough = await operator.mutate_single(
        idea,
        temperature=0.95,
        use_breakthrough=True
    )
    
    # Evaluate both
    regular_eval = await evaluator.evaluate(regular)
    breakthrough_eval = await evaluator.evaluate(breakthrough)
    
    return {
        "improvement_regular": regular_eval.overall_fitness - fitness,
        "improvement_breakthrough": breakthrough_eval.overall_fitness - fitness,
        "cost_regular": regular.cost,
        "cost_breakthrough": breakthrough.cost
    }
```

## Next Steps

1. Review and approve this plan
2. Create feature branch: `feature/diversity-breakthrough-enhancements`
3. Start with Phase 2 (name unification) as quick win
4. Implement Phase 1 (semantic diversity) for immediate impact
5. Validate Phase 3 before proceeding to Phase 4