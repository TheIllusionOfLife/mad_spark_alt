# Mad Spark Alt User Experience Guide

## Overview

Mad Spark Alt is a Multi-Agent Idea Generation System based on the QADI (Question → Abduction → Deduction → Induction) methodology from "Shin Logical Thinking". The system uses specialized AI agents to explore problems from multiple cognitive perspectives and generate comprehensive solution spaces.

## 🚨 Important: Current Implementation Status

**This is Phase 1 - Template-Based Implementation**

The current system demonstrates the complete QADI framework with **template-based generation**. This means:

✅ **What Works**: All 4 QADI phases run successfully with structured templates  
✅ **Framework**: Complete orchestration, registry, and agent coordination  
✅ **Output**: Consistent, structured ideas following QADI methodology  

🔍 **Expected Results**: The system generates template-based ideas like:
- "What are the core elements of [your problem]?"
- "What if [your problem] is caused by unexpected interactions?"  
- "If we address [your problem], then logically we must consider..."
- "Looking at patterns in [your problem], I observe recurring themes..."

**This is the intended Phase 1 behavior** - proving the framework architecture works with predefined templates before adding AI-powered reasoning in future phases.

## Quick Start

### Prerequisites
```bash
# Clone the repository
git clone https://github.com/TheIllusionOfLife/mad_spark_alt.git
cd mad_spark_alt

# Install dependencies
uv sync

# Install the package in development mode
uv pip install -e .
```

## Current System Capabilities

### 1. CLI Evaluation System

The CLI currently supports creativity evaluation commands:

#### Basic Evaluation
```bash
# Evaluate a single text for creativity
uv run mad-spark evaluate "The AI dreamed of electric sheep in quantum meadows, where digital butterflies computed the meaning of existence."

# Evaluate with verbose output
uv run mad-spark evaluate "Innovative blockchain-based social media platform" --verbose
```

#### Batch Evaluation
```bash
# Create test files
echo "A revolutionary app that uses AI to match people with their perfect houseplants" > idea1.txt
echo "Smart mirrors that provide personalized compliments and motivation" > idea2.txt
echo "A subscription service for renting different pets for a week" > idea3.txt

# Evaluate multiple files
uv run mad-spark batch-evaluate idea1.txt idea2.txt idea3.txt
```

#### Compare Multiple Ideas
```bash
# Compare creativity of different responses
uv run mad-spark compare "Traditional email marketing" "AI-powered personalized video messages" "Holographic business cards"
```

#### List Available Evaluators
```bash
# See what evaluation methods are available
uv run mad-spark list-evaluators
```

### 2. QADI Generation System (Python API)

The core QADI generation system is available through the Python API:

#### Complete QADI Cycle
```python
# Create a test script: test_qadi.py
import asyncio
from mad_spark_alt.core.orchestrator import QADIOrchestrator
from mad_spark_alt.core.interfaces import IdeaGenerationRequest

async def main():
    orchestrator = QADIOrchestrator()
    
    request = IdeaGenerationRequest(
        problem_statement="How might we reduce food waste in restaurants?",
        max_ideas_per_method=5
    )
    
    result = await orchestrator.generate_ideas(request)
    
    print(f"Generated {len(result.all_ideas)} ideas across {len(result.phases)} phases:")
    
    for phase_name, phase_result in result.phases.items():
        print(f"\n{phase_name.upper()} PHASE ({len(phase_result.generated_ideas)} ideas):")
        for idea in phase_result.generated_ideas:
            print(f"  💡 {idea.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
uv run python test_qadi.py
```

#### Individual Agent Testing
```python
# Create individual agent test: test_agents.py
import asyncio
from mad_spark_alt.agents import QuestioningAgent, AbductionAgent, DeductionAgent, InductionAgent
from mad_spark_alt.core.interfaces import IdeaGenerationRequest

async def test_agent(agent, agent_name, problem):
    print(f"\n{agent_name}:")
    request = IdeaGenerationRequest(problem_statement=problem, max_ideas_per_method=3)
    result = await agent.generate_ideas(request)
    for idea in result.generated_ideas:
        print(f"  • {idea.content}")

async def main():
    problem = "How can we make online learning more engaging?"
    
    # Test all four agents
    agents = [
        (QuestioningAgent(), "🤔 QUESTIONING AGENT"),
        (AbductionAgent(), "💡 ABDUCTION AGENT"),
        (DeductionAgent(), "🔍 DEDUCTION AGENT"),
        (InductionAgent(), "🔗 INDUCTION AGENT")
    ]
    
    for agent, name in agents:
        await test_agent(agent, name, problem)

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
uv run python test_agents.py
```

## Example Problem Domains

### Business Innovation
```python
# Create business_innovation.py
import asyncio
from mad_spark_alt.core.orchestrator import QADIOrchestrator
from mad_spark_alt.core.interfaces import IdeaGenerationRequest

async def main():
    orchestrator = QADIOrchestrator()
    
    problems = [
        "How might we create a subscription service that customers actually love?",
        "What new business model could disrupt the traditional retail industry?",
        "How can we make remote work more productive and engaging?"
    ]
    
    for problem in problems:
        print(f"\n{'='*60}")
        print(f"PROBLEM: {problem}")
        print('='*60)
        
        request = IdeaGenerationRequest(
            problem_statement=problem,
            max_ideas_per_method=3
        )
        
        result = await orchestrator.generate_ideas(request)
        
        for phase_name, phase_result in result.phases.items():
            print(f"\n{phase_name.upper()}:")
            for idea in phase_result.generated_ideas:
                print(f"  • {idea.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Educational Challenges
```python
# Create education_problems.py
import asyncio
from mad_spark_alt.core.orchestrator import QADIOrchestrator
from mad_spark_alt.core.interfaces import IdeaGenerationRequest

async def main():
    orchestrator = QADIOrchestrator()
    
    request = IdeaGenerationRequest(
        problem_statement="How can we make mathematics more engaging for middle school students?",
        max_ideas_per_method=4,
        context="Focus on practical applications and interactive learning"
    )
    
    result = await orchestrator.generate_ideas(request)
    
    print("QADI ANALYSIS: Making Mathematics Engaging")
    print("="*50)
    
    for phase_name, phase_result in result.phases.items():
        print(f"\n{phase_name.upper()} PHASE:")
        for i, idea in enumerate(phase_result.generated_ideas, 1):
            print(f"  {i}. {idea.content}")
            if hasattr(idea, 'reasoning') and idea.reasoning:
                print(f"     Reasoning: {idea.reasoning}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Technical Problem Solving
```python
# Create technical_problems.py
import asyncio
from mad_spark_alt.core.orchestrator import QADIOrchestrator
from mad_spark_alt.core.interfaces import IdeaGenerationRequest

async def main():
    orchestrator = QADIOrchestrator()
    
    request = IdeaGenerationRequest(
        problem_statement="How might we reduce the carbon footprint of data centers?",
        max_ideas_per_method=5,
        context="Consider current technology constraints and economic viability"
    )
    
    result = await orchestrator.generate_ideas(request)
    
    print("TECHNICAL ANALYSIS: Data Center Carbon Footprint")
    print("="*55)
    
    # Show summary first
    print(f"\nSUMMARY:")
    print(f"Total Ideas Generated: {len(result.all_ideas)}")
    print(f"Phases Completed: {len(result.phases)}")
    
    # Show detailed breakdown
    for phase_name, phase_result in result.phases.items():
        print(f"\n{phase_name.upper()} PHASE ({len(phase_result.generated_ideas)} ideas):")
        for idea in phase_result.generated_ideas:
            print(f"  💡 {idea.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Demo Scripts

### Run Provided Examples
```bash
# Run the comprehensive QADI demo
uv run python examples/qadi_demo.py

# Run basic usage examples
uv run python examples/basic_usage.py
```

## Advanced Usage Patterns

### Custom Configuration
```python
# Create custom_config.py
import asyncio
from mad_spark_alt.core.orchestrator import QADIOrchestrator
from mad_spark_alt.core.interfaces import IdeaGenerationRequest

async def main():
    orchestrator = QADIOrchestrator()
    
    # Custom configuration with specific parameters
    request = IdeaGenerationRequest(
        problem_statement="How can we create more sustainable packaging solutions?",
        max_ideas_per_method=8,
        context="Focus on biodegradable materials and circular economy principles",
        generation_config={
            "creativity_level": "high",
            "explore_analogies": True,
            "question_types": ["what", "why", "how", "assumptions"]
        }
    )
    
    result = await orchestrator.generate_ideas(request)
    
    print("SUSTAINABLE PACKAGING SOLUTIONS")
    print("="*35)
    
    for phase_name, phase_result in result.phases.items():
        print(f"\n{phase_name.upper()}:")
        for idea in phase_result.generated_ideas:
            print(f"  • {idea.content}")
            # Show confidence if available
            if hasattr(idea, 'confidence_score'):
                print(f"    Confidence: {idea.confidence_score:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Error Handling and Robustness
```python
# Create robust_usage.py
import asyncio
import logging
from mad_spark_alt.core.orchestrator import QADIOrchestrator
from mad_spark_alt.core.interfaces import IdeaGenerationRequest

# Enable logging to see what's happening
logging.basicConfig(level=logging.INFO)

async def main():
    orchestrator = QADIOrchestrator()
    
    try:
        request = IdeaGenerationRequest(
            problem_statement="How might we solve traffic congestion in megacities?",
            max_ideas_per_method=6
        )
        
        result = await orchestrator.generate_ideas(request)
        
        print("TRAFFIC CONGESTION SOLUTIONS")
        print("="*30)
        
        if result.phases:
            for phase_name, phase_result in result.phases.items():
                if phase_result.generated_ideas:
                    print(f"\n{phase_name.upper()}:")
                    for idea in phase_result.generated_ideas:
                        print(f"  • {idea.content}")
                else:
                    print(f"\n{phase_name.upper()}: No ideas generated")
        else:
            print("No phases completed successfully")
            
    except Exception as e:
        print(f"Error during generation: {e}")
        logging.error(f"Generation failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Understanding the Output

The QADI system generates structured results with:

- **Questioning Phase**: Explores assumptions, clarifies the problem, identifies constraints
- **Abduction Phase**: Generates creative hypotheses and explores analogies
- **Deduction Phase**: Applies logical reasoning and analyzes consequences
- **Induction Phase**: Synthesizes patterns and generalizes insights

Each idea includes:
- Content (the actual idea)
- Confidence score (when available)
- Reasoning explanation
- Metadata about generation approach

## Testing and Validation

### Run Tests
```bash
# Run the complete test suite
uv run pytest

# Run specific QADI system tests
uv run pytest tests/test_qadi_system.py -v

# Run with coverage
uv run pytest --cov=src/mad_spark_alt --cov-report=html
```

### Validate Setup
```bash
# Check if all components are properly registered
python -c "
from mad_spark_alt.core.registry import agent_registry
print('Available agents:', [agent.name for agent in agent_registry.get_agents()])
"
```

## Tips for Best Results

1. **Be Specific**: Instead of "How to improve customer service" use "How to reduce customer service response time for SaaS companies"

2. **Provide Context**: Use the `context` parameter to give agents relevant background information

3. **Experiment with Idea Counts**: Try different values for `max_ideas_per_method` (3-10 works well)

4. **Combine Evaluation**: Use the CLI evaluation system to assess the creativity of generated ideas

5. **Use Progressive Refinement**: Start with broad problems, then use specific agents for deeper exploration

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you've installed the package with `uv pip install -e .`

2. **No Ideas Generated**: Check that agents are properly registered and no exceptions are being silently caught

3. **Performance Issues**: For large idea counts, consider reducing `max_ideas_per_method`

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Then run your QADI generation
```

## Next Steps

This system provides a foundation for systematic idea generation. You can:

1. **Extend Agents**: Add new thinking method agents with specialized approaches
2. **Customize Templates**: Modify the idea generation templates for domain-specific problems
3. **Integration**: Combine with external APIs, databases, or knowledge sources
4. **Evaluation**: Use the built-in creativity evaluation system to assess generated ideas
5. **Human-AI Collaboration**: Build interfaces for interactive ideation sessions

The QADI methodology transforms problem-solving by applying systematic thinking methods through AI agents, helping you explore problems from multiple angles and generate comprehensive solution spaces.