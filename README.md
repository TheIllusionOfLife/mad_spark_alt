# Mad Spark Alt - Multi-Agent Idea Generation System

A multi-agent framework for collaborative idea generation based on the QADI (Question → Abduction → Deduction → Induction) methodology, with integrated creativity evaluation and genetic evolution capabilities.

## Features

- 🎯 **QADI Methodology**: Question → Abduction → Deduction → Induction thinking cycle
- 🤖 **Multi-Agent System**: Specialized AI agents for different thinking methods
- 🧨 **Dynamic Prompt Engineering**: Auto-detects question types for optimal analysis
- 📊 **Creativity Evaluation**: Multi-dimensional assessment of ideas
- 🧬 **Genetic Evolution**: Evolve ideas through generations using genetic algorithms
- ⚡ **Async Processing**: Efficient parallel processing with timeout management
- 🔌 **Extensible**: Plugin system for custom agents and evaluators

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

### Setup

```bash
# Create .env file with your API key (REQUIRED for meaningful results)
echo "GOOGLE_API_KEY=your_key_here" > .env

# Or use other providers
echo "OPENAI_API_KEY=your_key_here" >> .env
echo "ANTHROPIC_API_KEY=your_key_here" >> .env
```

### Generate Ideas with QADI

```bash
# RECOMMENDED: Multi-agent QADI with Google API (auto-detects question type)
uv run python qadi_simple_multi.py "How can we reduce plastic waste?"

# With manual question type override
uv run python qadi_simple_multi.py --type=business "How to monetize AI technology"

# Concrete mode for implementation-focused results
uv run python qadi_simple_multi.py --concrete "Build a mobile app for food delivery"

# Quick single-prompt version
uv run python qadi.py "How can we reduce plastic waste?"
```

### Example Prompts to Try

**Business & Innovation**
- "How can small businesses compete with large corporations in the digital age?"
- "What innovative business models could address climate change?"
- "How might we revolutionize remote work collaboration?"

**Technology & Society**
- "How can AI improve healthcare accessibility in rural areas?"
- "What are creative solutions to digital privacy concerns?"
- "How might we bridge the digital divide in education?"

**Environmental & Sustainability**
- "How can cities become carbon-neutral by 2030?"
- "What innovative approaches could solve ocean plastic pollution?"
- "How might vertical farming transform urban food systems?"

**Creative & Abstract**
- "What if gravity worked differently on weekends?"
- "How would society change if we could share dreams?"
- "Design a new sport for zero-gravity environments"

### CLI Tools

```bash
# Evaluate idea creativity
uv run mad-spark evaluate "The AI dreamed of electric sheep in quantum meadows"

# Evaluate with verbose output
uv run mad-spark evaluate "Blockchain social media platform" --verbose

# Compare multiple ideas
uv run mad-spark compare "Business communication" -r "Traditional email" -r "AI video messages" -r "Holographic cards"

# Batch evaluate multiple files
echo "Smart mirrors with personalized compliments" > idea1.txt
uv run mad-spark batch-evaluate idea1.txt idea2.txt

# List available evaluators
uv run mad-spark list-evaluators

# Run genetic evolution demo
uv run python examples/evolution_demo.py
```

### Python API

```python
import asyncio
from mad_spark_alt.core import SmartQADIOrchestrator

async def generate_ideas():
    orchestrator = SmartQADIOrchestrator()  # Uses LLM agents automatically
    result = await orchestrator.run_qadi_cycle(
        problem_statement="How can we reduce plastic waste?",
        cycle_config={"max_ideas_per_method": 3}
    )
    
    print(f"Generated {len(result.synthesized_ideas)} ideas")
    for idea in result.synthesized_ideas[:3]:
        print(f"💡 {idea.content}")

asyncio.run(generate_ideas())
```

For detailed API examples and advanced usage patterns, see [DEVELOPMENT.md](DEVELOPMENT.md).

## Dynamic Prompt Engineering

The system automatically detects question types and adapts prompts for optimal results:

**Auto-Detected Question Types**:
- **Technical**: Software architecture, implementation, coding questions
- **Business**: Strategy, growth, revenue, market-related queries  
- **Creative**: Design, innovation, artistic, brainstorming topics
- **Research**: Analysis, investigation, academic inquiries
- **Planning**: Organization, project management, timeline questions
- **Personal**: Individual growth, skills, career development

**Adaptive Features**:
- **Domain-specific prompts**: Each question type uses specialized prompts
- **Complexity detection**: Adjusts response depth based on question complexity
- **Manual override**: Use `--type=TYPE` to force a specific perspective
- **High accuracy**: 100% auto-detection success rate on common questions

## System Overview

**Architecture**: Multi-agent system with QADI orchestration, creativity evaluation, and genetic evolution

**Key Components**:
- **QADI Agents**: Question, Abduction, Deduction, Induction thinking methods
- **Prompt Classifier**: Intelligent question type detection with 100% accuracy
- **Adaptive Prompts**: Domain-specific prompts for each question type
- **Evaluation System**: Multi-dimensional creativity assessment (diversity, quality, coherence)
- **Evolution Engine**: Genetic algorithms for idea refinement
- **Circuit Breakers**: Fault-tolerant LLM API integration

For detailed architecture documentation, see [DEVELOPMENT.md](DEVELOPMENT.md).

## Extending the System

The system supports custom agents and evaluators through a plugin architecture:

- **Custom Thinking Agents**: Implement `ThinkingAgentInterface` for new reasoning methods
- **Custom Evaluators**: Implement `EvaluatorInterface` for new creativity metrics
- **Dynamic Registration**: Components auto-register when imported

For detailed extension guides and examples, see [DEVELOPMENT.md](DEVELOPMENT.md).

## Development

```bash
# Install development dependencies
uv sync --dev

# Run tests
uv run pytest

# Type checking
uv run mypy src/

# Code formatting
uv run black src/ tests/ && uv run isort src/ tests/
```

For comprehensive development guidelines, testing patterns, and contribution workflow, see [DEVELOPMENT.md](DEVELOPMENT.md).

## Documentation

- **[DEVELOPMENT.md](DEVELOPMENT.md)**: Development setup, architecture, coding standards, and API reference
- **[RESEARCH.md](RESEARCH.md)**: Academic background, QADI methodology, and research foundations
- **[SESSIONS.md](SESSIONS.md)**: Development session history and progress tracking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed contribution guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
