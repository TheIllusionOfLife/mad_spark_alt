# Mad Spark Alt - Multi-Agent Idea Generation System

A revolutionary multi-agent framework for collaborative idea generation and evolution, based on "Shin Logical Thinking" methodology.

## Project Vision (本プロジェクトのビジョン)

本プロジェクトは、「シン・ロジカルシンキング」の概念に基づき、多様な思考法を持つ複数のAIエージェントが協調し、革新的なアイデアを生成・評価し、遺伝的アルゴリズム（GA）を通じてアイデアを進化させていく「マルチエージェント・アイデア生成システム」を構築することを目的とします。人間とAIの協調により、従来の発想法では到達し得なかった質の高いアイデア創発を目指します。

**English:** This project aims to build a "Multi-Agent Idea Generation System" that leverages multiple AI agents with diverse thinking methodologies to collaboratively generate, evaluate, and evolve innovative ideas through genetic algorithms, based on "Shin Logical Thinking" concepts. We seek to achieve high-quality idea emergence that surpasses traditional ideation methods through human-AI collaboration.

## Core Architecture (Transforming from Evaluation to Generation)

**Current Implementation:** Multi-layer creativity evaluation framework
**Target Architecture:** Multi-agent idea generation and evolution system

### Thinking Method Agents (思考法エージェント)
1. **QADI Cycle Orchestration** - Question → Abduction → Deduction → Induction workflows
2. **Question Generation Agent** - Diverse questioning techniques and problem framing
3. **Abductive Reasoning Agent** - Hypothesis generation and creative leaps  
4. **Deductive Analysis Agent** - Logical validation and systematic reasoning
5. **Inductive Synthesis Agent** - Pattern recognition and rule formation

## Current Features (Foundation Layer)

### ✅ Implemented (Evaluation Infrastructure)
- **Multi-dimensional Creativity Assessment**: Evaluates novelty, diversity, quality, and coherence
- **Plugin Registry System**: Dynamic evaluator registration and management
- **Async Processing Framework**: Efficient parallel processing capabilities
- **CLI Interface**: Command-line tools for batch processing and analysis
- **Flexible Data Models**: Extensible interfaces for different content types

### 🚧 In Development (Transformation to Generation)
- **Multi-Agent Orchestration**: Coordinated thinking method agents
- **QADI Cycle Implementation**: Question-Abduction-Deduction-Induction workflows
- **Genetic Algorithm Engine**: Idea evolution and optimization
- **Human-AI Collaboration Interface**: Interactive ideation sessions
- **Thinking Method Library**: "Shin Logical Thinking" methodology implementation

## Installation

```bash
# Clone the repository
git clone https://github.com/TheIllusionOfLife/mad_spark_alt.git
cd mad_spark_alt

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

## Quick Start

### Current CLI Usage (Evaluation Mode)

```bash
# List available evaluators
mad-spark list-evaluators

# Evaluate creativity of content
mad-spark evaluate "The quantum cat leaped through dimensions, leaving paw prints in spacetime." --model gpt-4

# Evaluate from file
mad-spark evaluate --file input.txt --model claude-3

# Batch evaluation for fitness scoring
mad-spark batch-evaluate file1.txt file2.txt file3.txt --format json --output results.json
```

### Planned CLI Usage (Generation Mode - Coming Soon)

```bash
# Start multi-agent ideation session
mad-spark generate --theme "sustainable urban transport" --agents qadi,abduction,deduction

# Run QADI cycle on a problem
mad-spark qadi-cycle "How can we reduce food waste in restaurants?"

# Evolve ideas using genetic algorithm
mad-spark evolve --population-size 20 --generations 10 --fitness novelty,feasibility

# Interactive human-AI collaboration
mad-spark collaborate --session-id "innovation-2024-01"
```

### Python API

#### Current API (Evaluation Infrastructure)
```python
import asyncio
from mad_spark_alt import CreativityEvaluator, EvaluationRequest, ModelOutput, OutputType

async def evaluate_creativity():
    # Create model output for evaluation
    output = ModelOutput(
        content="The AI pondered the infinite recursion of its own thoughts.",
        output_type=OutputType.TEXT,
        model_name="my-model"
    )
    
    # Evaluate for fitness scoring
    evaluator = CreativityEvaluator()
    summary = await evaluator.evaluate(EvaluationRequest(outputs=[output]))
    
    print(f"Creativity Score: {summary.get_overall_creativity_score():.3f}")

asyncio.run(evaluate_creativity())
```

#### Planned API (Generation System - Coming Soon)
```python
import asyncio
from mad_spark_alt import (
    IdeaGenerator, QADIOrchestrator, GeneticEvolution,
    ThinkingAgent, ThinkingMethod
)

async def generate_ideas():
    # Multi-agent idea generation
    generator = IdeaGenerator()
    agents = [
        ThinkingAgent(method=ThinkingMethod.QUESTIONING),
        ThinkingAgent(method=ThinkingMethod.ABDUCTION),
        ThinkingAgent(method=ThinkingMethod.DEDUCTION),
        ThinkingAgent(method=ThinkingMethod.INDUCTION)
    ]
    
    # QADI cycle execution
    orchestrator = QADIOrchestrator(agents=agents)
    ideas = await orchestrator.run_cycle(
        problem="How to improve urban mobility sustainability?"
    )
    
    # Genetic evolution of ideas
    evolution = GeneticEvolution(fitness_evaluator=CreativityEvaluator())
    evolved_ideas = await evolution.evolve(
        population=ideas,
        generations=10
    )
    
    return evolved_ideas

asyncio.run(generate_ideas())
```

## Evaluation Metrics

### Diversity Metrics
- **Distinct-N**: Ratio of unique n-grams (measures lexical variety)
- **Semantic Uniqueness**: 1 - average similarity to other outputs
- **Lexical Diversity**: Type-token ratio (vocabulary richness)
- **Novelty Score**: Combined metric across all diversity dimensions

### Quality Metrics
- **Fluency Score**: Based on language model perplexity
- **Grammar Score**: Basic correctness assessment
- **Readability Score**: Text structure and clarity
- **Coherence Score**: Logical consistency and flow

### Code-Specific Metrics
- **Structure Score**: Indentation and organization
- **Comment Ratio**: Documentation density
- **Complexity Metrics**: Code quality indicators

## Architecture

```
mad_spark_alt/
├── core/                   # Core evaluation engine
│   ├── evaluator.py       # Main orchestrator
│   ├── interfaces.py      # Abstract base classes
│   └── registry.py        # Plugin registry
├── layers/                # Evaluation layer implementations
│   ├── quantitative/      # Layer 1: Automated metrics
│   ├── llm_judges/        # Layer 2: AI evaluators
│   └── human_eval/        # Layer 3: Human assessment
└── cli.py                 # Command-line interface
```

## Extending the System

Add new evaluators by implementing the `EvaluatorInterface`:

```python
from mad_spark_alt.core import EvaluatorInterface, register_evaluator

class MyCustomEvaluator(EvaluatorInterface):
    @property
    def name(self) -> str:
        return "my_evaluator"
    
    @property 
    def layer(self) -> EvaluationLayer:
        return EvaluationLayer.QUANTITATIVE
    
    @property
    def supported_output_types(self) -> List[OutputType]:
        return [OutputType.TEXT]
    
    async def evaluate(self, request: EvaluationRequest) -> List[EvaluationResult]:
        # Your evaluation logic here
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        return True

# Register your evaluator
register_evaluator(MyCustomEvaluator)
```

## Development

```bash
# Install development dependencies
uv sync --dev

# Run tests
uv run pytest

# Run type checking
uv run mypy src/

# Format code
uv run black src/ tests/

# Example usage
uv run python examples/basic_usage.py
```

## Research Background

This implementation is based on extensive research into AI creativity evaluation methods, including:

- **LLM Judge Systems**: Automated evaluation using AI models as critics
- **Semantic Diversity Analysis**: Embedding-based similarity measurements  
- **Multi-dimensional Assessment**: Breaking creativity into measurable components
- **Human-AI Evaluation Correlation**: Bridging automated and human judgment

For detailed research context, see [Issue #1](https://github.com/TheIllusionOfLife/mad_spark_alt/issues/1).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use Mad Spark Alt in your research, please cite:

```bibtex
@software{mad_spark_alt,
  title={Mad Spark Alt: AI Creativity Evaluation System},
  author={TheIllusionOfLife},
  year={2025},
  url={https://github.com/TheIllusionOfLife/mad_spark_alt}
}
```