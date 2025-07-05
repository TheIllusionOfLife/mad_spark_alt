# Hybrid Multi-layer Evaluation Framework

## Overview

Mad Spark Alt now implements a comprehensive **Hybrid Multi-layer Evaluation Framework** for AI creativity assessment, combining automated metrics, AI-powered evaluation, and human judgment for robust and reliable creativity measurement.

## Three-Layer Architecture

### Layer 1: Quantitative Automated Scanning ✅
**Purpose**: Large-scale filtering and basic quality/diversity assessment

**Evaluators**:
- `DiversityEvaluator`: Measures novelty through distinct n-grams, semantic distance, lexical diversity
- `QualityEvaluator`: Assesses fluency, grammar, readability, coherence

**Usage**:
```bash
# Basic quantitative evaluation
mad-spark evaluate "your creative text" --layers quantitative
```

### Layer 2: LLM-based Contextual Evaluation ✅
**Purpose**: Scalable quality and contextual assessment using AI models

**Evaluators**:
- `CreativityLLMJudge`: Single AI model evaluation with decomposed creativity axes
- `CreativityJury`: Multi-judge consensus system with disagreement detection

**Features**:
- **Decomposed Assessment**: Evaluates novelty, usefulness, feasibility, elaboration, surprise, elegance
- **Transparent Reasoning**: Structured JSON output with rationales and explanations
- **Multiple Model Support**: OpenAI GPT, Anthropic Claude, and mock models for testing
- **Consensus Mechanisms**: Voting and disagreement detection across multiple judges
- **Cost Tracking**: Token usage monitoring and budget management

**Usage**:
```bash
# Single LLM judge evaluation
mad-spark llm-judge "your creative text" --model gpt-4

# Multi-judge jury evaluation
mad-spark llm-jury "your creative text" --models gpt-4,claude-3-sonnet

# Specify original prompt for context
mad-spark llm-judge "creative response" --prompt "original prompt" --model claude-3-sonnet
```

### Layer 3: Human Assessment Interfaces ✅
**Purpose**: Expert and target user evaluation for subjective creativity assessment

**Evaluators**:
- `HumanCreativityEvaluator`: Structured human assessment interface
- `ABTestEvaluator`: Comparative evaluation between multiple outputs

**Modes**:
- **Interactive**: Real-time human evaluation with guided prompts
- **Batch**: Generate evaluation templates for offline assessment
- **Expert**: Load and process completed expert evaluations
- **A/B Testing**: Pairwise, ranking, or tournament comparisons

**Usage**:
```bash
# Interactive human evaluation
mad-spark human-eval "your creative text" --mode interactive

# Generate batch evaluation template
mad-spark human-eval "your creative text" --mode batch --output evaluations.json

# Load expert evaluation results
mad-spark human-eval "your creative text" --mode expert --input completed_evals.json

# A/B test comparison
mad-spark ab-test --texts "option 1" "option 2" "option 3" --mode pairwise
```

## Research Foundation

This implementation is based on extensive creativity evaluation research:

### LLM-as-Judge Methodology
- **Decomposed Evaluation**: Breaking creativity into measurable dimensions
- **Few-shot Prompting**: Transparent scoring with detailed rationales
- **Multi-judge Consensus**: Reducing individual model biases through voting mechanisms
- **Benchmark Comparison**: Evaluating against human-created gold standards

### Human-AI Evaluation Correlation
- **Structured Assessment**: Standardized rating scales (1-10) mapped to 0-1 scores
- **Expert vs. User Evaluation**: Different perspectives for comprehensive assessment
- **A/B Testing**: Direct comparative evaluation for relative creativity measurement

## Complete Framework Usage

### CLI Commands

```bash
# List all available evaluators (all layers)
mad-spark list-evaluators

# Traditional quantitative evaluation
mad-spark evaluate "text" --layers quantitative

# LLM judge evaluation
mad-spark llm-judge "text" --model gpt-4
mad-spark llm-jury "text" --models gpt-4,claude-3-sonnet

# Human evaluation
mad-spark human-eval "text" --mode interactive
mad-spark ab-test --texts "option1" "option2" --mode ranking

# Combined evaluation (multiple layers)
mad-spark evaluate "text" --layers quantitative,llm_judge
```

### Programmatic Usage

```python
import asyncio
from mad_spark_alt.core import EvaluationRequest, EvaluationLayer, ModelOutput, OutputType
from mad_spark_alt.layers.llm_judges import CreativityLLMJudge, CreativityJury
from mad_spark_alt.layers.human_eval import HumanCreativityEvaluator

# Layer 2: LLM Judge Evaluation
async def llm_evaluation():
    judge = CreativityLLMJudge("gpt-4")
    
    output = ModelOutput(
        content="Your creative content",
        output_type=OutputType.TEXT,
        model_name="your-model",
        prompt="Original prompt"
    )
    
    request = EvaluationRequest(
        outputs=[output],
        target_layers=[EvaluationLayer.LLM_JUDGE]
    )
    
    results = await judge.evaluate(request)
    return results

# Layer 2: Multi-judge Jury
async def jury_evaluation():
    jury = CreativityJury(["gpt-4", "claude-3-sonnet"])
    # ... same request setup
    results = await jury.evaluate(request)
    return results

# Layer 3: Human Evaluation
async def human_evaluation():
    evaluator = HumanCreativityEvaluator({"mode": "batch"})
    # ... same request setup
    results = await evaluator.evaluate(request)
    return results
```

## Configuration Options

### LLM Judge Configuration
```python
config = {
    "temperature": 0.0,          # Deterministic outputs
    "max_tokens": 1500,          # Response length limit
    "dimensions": ["novelty", "usefulness"],  # Specific dimensions
    "include_rationale": True,   # Detailed explanations
}

judge = CreativityLLMJudge("gpt-4", config)
```

### Jury Configuration
```python
config = {
    "consensus_method": "median",     # "median", "mean"
    "disagreement_threshold": 0.3,   # Disagreement detection
    "min_agreement": 0.7,           # Minimum consensus requirement
}

jury = CreativityJury(["gpt-4", "claude-3"], config)
```

### Human Evaluation Configuration
```python
config = {
    "mode": "interactive",       # "interactive", "batch", "expert"
    "output_file": "evals.json", # For batch mode
    "input_file": "completed.json", # For expert mode
    "rating_scale": "1-10",      # Rating scale
}

evaluator = HumanCreativityEvaluator(config)
```

## Output Format

### LLM Judge Results
```json
{
  "creativity_scores": {
    "novelty": 0.85,
    "usefulness": 0.75,
    "feasibility": 0.90,
    "elaboration": 0.80,
    "surprise": 0.70,
    "elegance": 0.85,
    "overall_creativity": 0.81
  },
  "rationale": "Detailed explanation of assessment...",
  "strengths": ["Original concept", "Clear execution"],
  "weaknesses": ["Could be more detailed", "Limited scope"],
  "metadata": {
    "model": "gpt-4",
    "usage": {"total_tokens": 1250},
    "evaluation_dimensions": ["novelty", "usefulness", ...]
  }
}
```

### Jury Consensus Results
```json
{
  "creativity_scores": {
    "novelty": 0.82,  // Median consensus
    "usefulness": 0.73,
    "overall_creativity": 0.79
  },
  "explanations": {
    "consensus_rationale": "Jury of 3 judges evaluated...",
    "consensus_quality": "High agreement among judges",
    "disagreement_notice": "Significant disagreement on: feasibility"
  },
  "metadata": {
    "jury_size": 3,
    "consensus_method": "median",
    "disagreement_analysis": {...},
    "models": ["gpt-4", "claude-3-sonnet", "gemini-pro"]
  }
}
```

### Human Evaluation Results
```json
{
  "scores": {
    "novelty": 0.78,         // Converted from 1-10 scale
    "usefulness": 0.67,
    "feasibility": 0.89,
    "overall_creativity": 0.75
  },
  "explanations": {
    "human_comments": "Innovative concept but needs refinement",
    "evaluator_id": "expert_001",
    "evaluation_time": 120.5
  },
  "metadata": {
    "raw_ratings": {
      "novelty": 8,          // Original 1-10 ratings
      "usefulness": 7,
      "feasibility": 9,
      "overall_creativity": 8
    }
  }
}
```

## Implementation Benefits

### Architectural Strengths
- **Minimal Breaking Changes**: Existing evaluation APIs remain fully functional
- **Plugin Architecture**: Dynamic evaluator management and registration
- **Async Patterns**: Efficient parallel processing for multiple evaluations
- **Strong Foundation**: Extensible interfaces and data models

### Research-Backed Approach
- **Multi-dimensional Assessment**: Comprehensive creativity measurement
- **Consensus Mechanisms**: Improved reliability through multiple perspectives
- **Human-AI Correlation**: Bridging automated and human judgment
- **Scalable Evaluation**: From quick automated screening to detailed human assessment

### Practical Applications
- **Development/Experimentation**: Fast iteration with automated layers
- **Quality Assurance**: Reliable assessment through jury consensus
- **Final Validation**: Human expert evaluation for critical decisions
- **A/B Testing**: Direct comparison for product decisions

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -e .
   ```

2. **Set API Keys** (for real LLM evaluation):
   ```bash
   export OPENAI_API_KEY="your-key"
   export ANTHROPIC_API_KEY="your-key"
   ```

3. **Try the Framework**:
   ```bash
   # Start with mock models (no API keys needed)
   mad-spark llm-judge "your creative text" --model mock-model
   
   # Use real models when ready
   mad-spark llm-jury "your text" --models gpt-4,claude-3-sonnet
   ```

4. **Explore Human Evaluation**:
   ```bash
   # Interactive evaluation
   mad-spark human-eval "your text" --mode interactive
   
   # Generate batch templates
   mad-spark human-eval "your text" --mode batch
   ```

The hybrid framework provides a complete research-backed solution for AI creativity evaluation, supporting both rapid development workflows and rigorous assessment requirements.