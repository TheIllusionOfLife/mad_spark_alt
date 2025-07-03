# Mad Spark Alt Documentation

Welcome to the comprehensive documentation for Mad Spark Alt, a multi-agent idea generation system based on "Shin Logical Thinking" QADI methodology.

## 📚 Documentation Index

### 🚀 Getting Started
- **[Project README](../README.md)** - Main project overview, installation, and quick start
- **[Examples Guide](examples.md)** - Comprehensive usage examples and code samples
- **[CLI Usage Guide](cli_usage.md)** - Complete command-line interface documentation

### 🏗️ Development
- **[CLAUDE.md](../CLAUDE.md)** - Project instructions for AI development assistance
- **[QADI API Documentation](qadi_api.md)** - Complete API reference for all QADI components

### 🧪 Testing
- **Test Files**: Located in `tests/` directory
  - `test_qadi_system.py` - QADI system integration tests
  - `unit/` - Individual component unit tests

### 📁 Code Examples
- **Example Scripts**: Located in `examples/` directory
  - `qadi_demo.py` - Interactive QADI system demonstration
  - `basic_usage.py` - Basic evaluation and generation examples

## 🎯 QADI System Overview

The QADI (Question → Abduction → Deduction → Induction) system implements "Shin Logical Thinking" methodology through specialized AI agents:

### Core Components

1. **🤔 Questioning Agent** - Generates diverse questions and problem framings
2. **💡 Abduction Agent** - Creates hypotheses and makes creative leaps  
3. **🔍 Deduction Agent** - Performs logical validation and systematic reasoning
4. **🔗 Induction Agent** - Synthesizes patterns and forms general principles
5. **🎛️ QADI Orchestrator** - Coordinates multi-phase thinking cycles

### Key Features

- ✅ **Multi-Agent Coordination** - Seamless collaboration between thinking agents
- ✅ **Async Processing** - Efficient parallel and sequential processing
- ✅ **Error Resilience** - Robust handling of missing or failing components
- ✅ **Creativity Evaluation** - Integrated assessment of generated ideas
- ✅ **Extensible Architecture** - Plugin system for custom agents and evaluators

## 📖 Quick Navigation

### For New Users
1. Start with the [Project README](../README.md) for installation and basic concepts
2. Try the [Quick Start Examples](examples.md#quick-start-examples) 
3. Explore the [CLI Usage Guide](cli_usage.md) for command-line tools

### For Developers
1. Review the [QADI API Documentation](qadi_api.md) for complete technical reference
2. Check [CLAUDE.md](../CLAUDE.md) for development guidelines and architecture
3. Study the [Advanced Usage Examples](examples.md#advanced-usage-patterns)

### For Specific Use Cases
- **Business Innovation**: [Business Examples](examples.md#business-innovation)
- **Educational Technology**: [Education Examples](examples.md#educational-technology)  
- **Scientific Research**: [Research Examples](examples.md#scientific-research)
- **Creative Writing**: [CLI Creative Examples](cli_usage.md#creative-writing)

## 🔧 Development Workflow

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/TheIllusionOfLife/mad_spark_alt.git
cd mad_spark_alt

# Install with development dependencies
uv sync --dev

# Run tests to verify setup
uv run pytest tests/test_qadi_system.py -v

# Try the QADI demo
python examples/qadi_demo.py
```

### Running Examples

```bash
# Basic QADI demonstration
python examples/qadi_demo.py

# Basic evaluation examples  
python examples/basic_usage.py

# Test individual agents
python -c "
import asyncio
from mad_spark_alt.agents import QuestioningAgent
from mad_spark_alt.core import IdeaGenerationRequest

async def test():
    agent = QuestioningAgent()
    request = IdeaGenerationRequest(
        problem_statement='How can we improve urban sustainability?',
        max_ideas_per_method=3
    )
    result = await agent.generate_ideas(request)
    for idea in result.generated_ideas:
        print(f'💡 {idea.content}')

asyncio.run(test())
"
```

### Testing Strategy

```bash
# Run all tests
uv run pytest

# Run QADI system tests specifically
uv run pytest tests/test_qadi_system.py -v

# Run with coverage
uv run pytest --cov=src/mad_spark_alt --cov-report=html

# Type checking
uv run mypy src/

# Code formatting
uv run black src/ tests/
uv run isort src/ tests/
```

## 🏛️ Architecture Overview

```
mad_spark_alt/
├── 🧠 core/                        # Core system components
│   ├── orchestrator.py             # QADI cycle coordination
│   ├── interfaces.py               # Agent and evaluator interfaces
│   ├── registry.py                 # Unified component management
│   └── evaluator.py                # Creativity evaluation engine
├── 🤖 agents/                      # QADI thinking method agents
│   ├── questioning/                # Question generation
│   ├── abduction/                  # Hypothesis generation
│   ├── deduction/                  # Logical reasoning
│   └── induction/                  # Pattern synthesis
├── 📊 layers/                      # Evaluation infrastructure
│   ├── quantitative/               # Automated metrics
│   ├── llm_judges/                 # AI-powered evaluation
│   └── human_eval/                 # Human assessment
├── 📚 docs/                        # Documentation (this directory)
├── 🧪 tests/                       # Test suite
├── 📝 examples/                    # Usage examples
└── 🎮 cli.py                       # Command-line interface
```

## 🚀 Common Use Cases

### 1. Basic Idea Generation

```python
# Generate ideas for a problem
from mad_spark_alt.core import QADIOrchestrator

orchestrator = QADIOrchestrator()
result = await orchestrator.run_qadi_cycle(
    problem_statement="Your problem here",
    context="Additional context",
    cycle_config={"max_ideas_per_method": 3}
)
```

### 2. Creativity Evaluation

```python
# Evaluate generated content
from mad_spark_alt import CreativityEvaluator, EvaluationRequest, ModelOutput

evaluator = CreativityEvaluator()
summary = await evaluator.evaluate(
    EvaluationRequest(outputs=[ModelOutput(content="Your content")])
)
```

### 3. Custom Agent Development

```python
# Create custom thinking agent
from mad_spark_alt.core import ThinkingAgentInterface, register_agent

class MyAgent(ThinkingAgentInterface):
    # Implement required methods
    pass

register_agent(MyAgent)
```

## 📧 Support and Contributing

### Getting Help
- Check the [Examples](examples.md) for code samples
- Review [CLI Documentation](cli_usage.md) for command-line usage
- Consult [API Documentation](qadi_api.md) for technical details

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `uv run pytest`
5. Submit a pull request

### Reporting Issues
- Use GitHub Issues for bug reports and feature requests
- Include minimal reproducible examples
- Specify your environment and versions

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.

## 📊 Project Status

- ✅ **Core QADI System**: Fully implemented and tested
- ✅ **Thinking Agents**: All four agents (Q-A-D-I) complete
- ✅ **Evaluation System**: Multi-dimensional creativity assessment
- ✅ **Documentation**: Comprehensive guides and examples
- 🚧 **Genetic Evolution**: Planned for future releases
- 🚧 **Web Interface**: Planned for future releases

---

*This documentation is continuously updated. For the latest information, refer to the project repository and documentation files.*