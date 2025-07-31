# Hypothesis Format and Evolution Fixes

## Overview
This update addresses three key issues reported with the evolution system:
1. Redundant H+number prefix in hypothesis display (e.g., "Approach 1: H1: Title")
2. Debug log messages appearing in user output
3. Fallback text appearing instead of LLM-generated content

## Changes Made

### 1. Removed H+Number Prefix
- **Problem**: Hypotheses were displayed with redundant prefixes like "Approach 1: H1: Title"
- **Solution**: Updated prompts and parsing logic to use clean numbered format
- **Files Modified**:
  - `src/mad_spark_alt/core/qadi_prompts.py`: Changed prompt format from "H1:" to "1."
  - `src/mad_spark_alt/core/simple_qadi_orchestrator.py`: Updated deduction phase formatting
  - `qadi_simple.py`: Fixed display logic to show only "Approach N" without H prefix

### 2. Suppressed Evolution Debug Logs
- **Problem**: Debug messages like "Using fallback text for offspring..." appeared in output
- **Solution**: Changed log level from WARNING to DEBUG and configured logging during evolution
- **Files Modified**:
  - `src/mad_spark_alt/evolution/semantic_operators.py`: Changed logger.warning to logger.debug
  - `qadi_simple.py`: Added logging configuration to suppress DEBUG messages during evolution

### 3. Enhanced Structured Output Support
- **Problem**: Evolution operators sometimes produced fallback text instead of LLM content
- **Solution**: Added structured output support to single mutation operator
- **Files Modified**:
  - `src/mad_spark_alt/evolution/semantic_operators.py`: 
    - Added JSON schema for single mutations
    - Updated prompts to request JSON format
    - Added proper parsing with fallback

## Testing
All changes are covered by comprehensive tests:
- `tests/test_hypothesis_format.py`: Tests for H prefix removal
- `tests/test_evolution_logging.py`: Tests for log suppression
- `tests/test_semantic_operators_structured.py`: Tests for structured output

## Backward Compatibility
- Parser still handles old "H1:" format for compatibility
- Structured output includes fallback to text parsing
- No breaking changes to public APIs

## User Impact
- Cleaner output without redundant prefixes
- No more debug messages in evolution output
- More reliable LLM-generated variations in evolution
- Better structured output support reduces fallback text occurrences