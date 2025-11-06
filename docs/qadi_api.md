# QADI System API Documentation

## Overview

The QADI (Question → Abduction → Deduction → Induction) system implements "Shin Logical Thinking" methodology through a multi-agent framework. This document provides comprehensive API documentation for all QADI components.

## Core Interfaces

### ThinkingAgentInterface

Base interface for all thinking method agents.

```python
from abc import ABC, abstractmethod
from typing import List
from mad_spark_alt.core.interfaces import IdeaGenerationRequest, IdeaGenerationResult, ThinkingMethod, OutputType

class ThinkingAgentInterface(ABC):
    """Abstract base class for thinking method agents."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the agent's unique name."""
        pass
    
    @property
    @abstractmethod
    def thinking_method(self) -> ThinkingMethod:
        """Return the thinking method this agent implements."""
        pass
    
    @property
    @abstractmethod
    def supported_output_types(self) -> List[OutputType]:
        """Return list of supported output types."""
        pass
    
    @abstractmethod
    async def generate_ideas(self, request: IdeaGenerationRequest) -> IdeaGenerationResult:
        """Generate ideas based on the request."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate agent configuration."""
        pass
```

### Data Models

#### IdeaGenerationRequest

Input structure for idea generation requests.

```python
@dataclass
class IdeaGenerationRequest:
    """Request for idea generation from thinking agents."""
    
    problem_statement: str                                    # Main problem to address
    context: Optional[str] = None                            # Additional context
    target_thinking_methods: Optional[List[ThinkingMethod]] = None  # Specific methods
    max_ideas_per_method: int = 5                           # Limit ideas per agent
    require_reasoning: bool = False                         # Include reasoning
    generation_config: Optional[Dict[str, Any]] = None     # Agent-specific config
    
    def __post_init__(self):
        if self.target_thinking_methods is None:
            self.target_thinking_methods = list(ThinkingMethod)
        if self.generation_config is None:
            self.generation_config = {}
```

#### GeneratedIdea

Rich representation of a generated idea.

```python
@dataclass
class GeneratedIdea:
    """A single generated idea with metadata."""
    
    content: str                              # The idea content
    thinking_method: ThinkingMethod           # Method used to generate
    agent_name: str                          # Name of generating agent
    generation_prompt: str                   # Prompt that led to this idea
    confidence_score: float                  # Agent's confidence (0-1)
    reasoning: Optional[str] = None          # Explanation of reasoning
    metadata: Optional[Dict[str, Any]] = None # Additional metadata
    timestamp: Optional[str] = None          # Generation timestamp
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
```

#### IdeaGenerationResult

Output structure from thinking agents.

```python
@dataclass
class IdeaGenerationResult:
    """Result from a thinking agent's idea generation."""
    
    agent_name: str                                        # Name of the agent
    thinking_method: ThinkingMethod                        # Method used
    generated_ideas: List[GeneratedIdea]                   # Generated ideas
    execution_time: Optional[float] = None                 # Time taken
    error_message: Optional[str] = None                    # Error if any
    generation_metadata: Optional[Dict[str, Any]] = None   # Additional metadata
    
    def __post_init__(self):
        if self.generation_metadata is None:
            self.generation_metadata = {}
```

## QADI Thinking Agents

### QuestioningAgent

Generates diverse questions and problem framings.

#### Purpose
- **Primary Role**: Generate questions that explore different aspects of a problem
- **Thinking Focus**: Problem framing, assumption challenging, perspective shifting
- **Output**: Strategic questions that guide further thinking

#### Configuration Options

```python
config = {
    "question_types": ["clarifying", "alternative", "challenging", "expanding"],
    "max_questions_per_type": 2,
    "include_meta_questions": True,
    "focus_areas": ["assumptions", "constraints", "stakeholders", "outcomes"]
}
```

#### Usage Example

```python
import asyncio
from mad_spark_alt.agents import QuestioningAgent
from mad_spark_alt.core import IdeaGenerationRequest

async def use_questioning_agent():
    agent = QuestioningAgent()
    
    request = IdeaGenerationRequest(
        problem_statement="How can we reduce urban air pollution?",
        context="City with 2 million population, heavy traffic, industrial areas",
        max_ideas_per_method=4,
        require_reasoning=True,
        generation_config={
            "question_types": ["clarifying", "alternative"],
            "focus_areas": ["stakeholders", "constraints"]
        }
    )
    
    result = await agent.generate_ideas(request)
    
    for idea in result.generated_ideas:
        print(f"Question: {idea.content}")
        print(f"Reasoning: {idea.reasoning}")
        print(f"Strategy: {idea.metadata.get('strategy', 'general')}")
        print("---")

asyncio.run(use_questioning_agent())
```

#### Question Strategies

| Strategy | Description | Example Output |
|----------|-------------|----------------|
| **clarifying** | Clarify problem scope and definitions | "What specific pollutants are we targeting?" |
| **alternative** | Explore different problem framings | "What if we focused on pollution sources rather than effects?" |
| **challenging** | Challenge assumptions | "Are we assuming cars are the main problem?" |
| **expanding** | Broaden the problem scope | "How might this connect to broader urban planning?" |

### AbductionAgent

Generates creative hypotheses and makes intuitive leaps.

#### Purpose
- **Primary Role**: Generate creative hypotheses about underlying causes and patterns
- **Thinking Focus**: Creative leaps, pattern recognition, "what if" scenarios
- **Output**: Hypotheses that suggest novel explanations or approaches

#### Configuration Options

```python
config = {
    "hypothesis_types": ["causal", "analogical", "pattern", "opposite", "emergent"],
    "creativity_level": "balanced",  # "conservative", "balanced", "radical"
    "use_analogies": True,
    "explore_opposites": True,
    "pattern_sources": ["nature", "technology", "social_systems"]
}
```

#### Usage Example

```python
import asyncio
from mad_spark_alt.agents import AbductionAgent
from mad_spark_alt.core import IdeaGenerationRequest

async def use_abduction_agent():
    agent = AbductionAgent()
    
    request = IdeaGenerationRequest(
        problem_statement="How can we improve online learning engagement?",
        context="Remote education, students losing focus, need innovative approaches",
        max_ideas_per_method=3,
        require_reasoning=True,
        generation_config={
            "hypothesis_types": ["analogical", "emergent"],
            "creativity_level": "balanced",
            "use_analogies": True
        }
    )
    
    result = await agent.generate_ideas(request)
    
    for idea in result.generated_ideas:
        print(f"Hypothesis: {idea.content}")
        print(f"Type: {idea.metadata.get('strategy', 'creative_leap')}")
        print(f"Confidence: {idea.confidence_score:.2f}")
        print("---")

asyncio.run(use_abduction_agent())
```

#### Hypothesis Types

| Type | Description | Example Output |
|------|-------------|----------------|
| **causal** | Hidden cause hypotheses | "What if engagement issues stem from lack of social connection?" |
| **analogical** | Solutions from other domains | "What if online learning worked like video games?" |
| **pattern** | Pattern-based insights | "What if this follows the same pattern as habit formation?" |
| **opposite** | Counterintuitive approaches | "What if we made learning deliberately more challenging?" |
| **emergent** | Self-organizing solutions | "What if solutions emerge from student collaboration?" |

### DeductionAgent

Performs logical validation and systematic reasoning.

#### Purpose
- **Primary Role**: Apply logical reasoning to validate and explore implications
- **Thinking Focus**: Systematic analysis, constraint checking, logical consequences
- **Output**: Logical implications, constraints, and systematic analyses

#### Configuration Options

```python
config = {
    "reasoning_modes": ["logical", "constraint", "implication", "systematic"],
    "validation_depth": "thorough",  # "quick", "standard", "thorough"
    "include_constraints": True,
    "check_assumptions": True,
    "logical_frameworks": ["if_then", "necessary_sufficient", "cause_effect"]
}
```

#### Usage Example

```python
import asyncio
from mad_spark_alt.agents import DeductionAgent
from mad_spark_alt.core import IdeaGenerationRequest

async def use_deduction_agent():
    agent = DeductionAgent()
    
    request = IdeaGenerationRequest(
        problem_statement="Should our company adopt a 4-day work week?",
        context="Tech company, 200 employees, productivity concerns, employee satisfaction goals",
        max_ideas_per_method=4,
        require_reasoning=True,
        generation_config={
            "reasoning_modes": ["logical", "constraint"],
            "validation_depth": "thorough",
            "include_constraints": True
        }
    )
    
    result = await agent.generate_ideas(request)
    
    for idea in result.generated_ideas:
        print(f"Analysis: {idea.content}")
        print(f"Mode: {idea.metadata.get('reasoning_mode', 'logical')}")
        print(f"Reasoning: {idea.reasoning}")
        print("---")

asyncio.run(use_deduction_agent())
```

#### Reasoning Modes

| Mode | Description | Example Output |
|------|-------------|----------------|
| **logical** | Step-by-step logical analysis | "If productivity stays constant, then costs decrease by 20%" |
| **constraint** | Constraint and limitation analysis | "Implementation requires coordination with client schedules" |
| **implication** | Consequence exploration | "This would likely affect customer service availability" |
| **systematic** | Systematic framework application | "Using cost-benefit analysis: benefits outweigh costs if..." |

### InductionAgent

Synthesizes patterns and forms general principles.

#### Purpose
- **Primary Role**: Identify patterns and synthesize insights from available information
- **Thinking Focus**: Pattern recognition, generalization, principle formation
- **Output**: General insights, patterns, and synthesized principles

#### Configuration Options

```python
config = {
    "synthesis_methods": ["pattern", "principle", "insight", "generalization", "meta_pattern"],
    "pattern_depth": "deep",  # "surface", "moderate", "deep"
    "generalization_level": "broad",  # "narrow", "moderate", "broad"
    "insight_generation": True,
    "principle_extraction": True
}
```

#### Usage Example

```python
import asyncio
from mad_spark_alt.agents import InductionAgent
from mad_spark_alt.core import IdeaGenerationRequest

async def use_induction_agent():
    agent = InductionAgent()
    
    request = IdeaGenerationRequest(
        problem_statement="What makes successful remote teams?",
        context="Analysis of 50 remote teams, some successful, some struggling",
        max_ideas_per_method=3,
        require_reasoning=True,
        generation_config={
            "synthesis_methods": ["pattern", "principle"],
            "pattern_depth": "deep",
            "generalization_level": "broad"
        }
    )
    
    result = await agent.generate_ideas(request)
    
    for idea in result.generated_ideas:
        print(f"Insight: {idea.content}")
        print(f"Method: {idea.metadata.get('method', 'inductive_reasoning')}")
        print(f"Confidence: {idea.confidence_score:.2f}")
        print("---")

asyncio.run(use_induction_agent())
```

#### Synthesis Methods

| Method | Description | Example Output |
|--------|-------------|----------------|
| **pattern** | Pattern identification | "Successful teams show recurring communication patterns..." |
| **principle** | Principle extraction | "Core principle: Trust enables autonomy, autonomy enables performance" |
| **insight** | Key insight synthesis | "The most effective remote teams treat presence differently..." |
| **generalization** | Broad applicability | "These patterns apply to any distributed collaboration..." |
| **meta_pattern** | Higher-level patterns | "Teams exhibit self-organizing properties when conditions align..." |

## QADI Orchestration

### QADIOrchestrator

Coordinates the complete QADI thinking cycle.

#### Purpose
Manages the sequential execution of thinking methods in the QADI cycle, building enhanced context between phases and synthesizing final results.

#### Core Methods

```python
class QADIOrchestrator:
    """Orchestrates multi-agent QADI thinking cycles."""
    
    def __init__(self, agents: Optional[List[ThinkingAgentInterface]] = None):
        """Initialize with optional list of agents."""
        
    async def run_qadi_cycle(
        self,
        problem_statement: str,
        context: Optional[str] = None,
        cycle_config: Optional[Dict[str, Any]] = None
    ) -> QADICycleResult:
        """Run complete QADI cycle."""
        
    async def run_parallel_generation(
        self,
        problem_statement: str,
        thinking_methods: List[ThinkingMethod],
        context: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[ThinkingMethod, IdeaGenerationResult]:
        """Run multiple thinking methods in parallel."""
        
    def has_agent(self, method: ThinkingMethod) -> bool:
        """Check if agent for thinking method is available."""
        
    def get_agent(self, method: ThinkingMethod) -> Optional[ThinkingAgentInterface]:
        """Get agent for specific thinking method."""
```

#### Usage Example

```python
import asyncio
from mad_spark_alt.core import QADIOrchestrator, agent_registry, register_agent
from mad_spark_alt.agents import QuestioningAgent, AbductionAgent, DeductionAgent, InductionAgent

async def orchestrate_qadi_cycle():
    # Register all agents
    register_agent(QuestioningAgent)
    register_agent(AbductionAgent)
    register_agent(DeductionAgent)
    register_agent(InductionAgent)
    
    # Get agents from registry
    agents = [
        agent_registry.get_agent_by_method(method) 
        for method in [ThinkingMethod.QUESTIONING, ThinkingMethod.ABDUCTION, 
                      ThinkingMethod.DEDUCTION, ThinkingMethod.INDUCTION]
    ]
    
    # Create orchestrator
    orchestrator = QADIOrchestrator([a for a in agents if a])
    
    # Run complete QADI cycle
    result = await orchestrator.run_qadi_cycle(
        problem_statement="How can we create more sustainable cities?",
        context="Urban planning challenge, climate change considerations, citizen well-being",
        cycle_config={
            "max_ideas_per_method": 3,
            "require_reasoning": True,
            "creativity_level": "balanced"
        }
    )
    
    # Process results
    print(f"QADI Cycle completed in {result.execution_time:.2f}s")
    print(f"Cycle ID: {result.cycle_id}")
    
    for phase_name, phase_result in result.phases.items():
        print(f"\n{phase_name.title()} Phase:")
        print(f"  Agent: {phase_result.agent_name}")
        print(f"  Ideas: {len(phase_result.generated_ideas)}")
        
        for i, idea in enumerate(phase_result.generated_ideas, 1):
            print(f"    {i}. {idea.content}")
    
    print(f"\nSynthesized Ideas: {len(result.synthesized_ideas)}")

asyncio.run(orchestrate_qadi_cycle())
```

### QADICycleResult

Complete result from a QADI cycle execution.

```python
@dataclass
class QADICycleResult:
    """Result from a complete QADI cycle execution."""
    
    problem_statement: str                              # Original problem
    cycle_id: str                                      # Unique cycle identifier
    phases: Dict[str, IdeaGenerationResult]            # Results by phase
    synthesized_ideas: List[GeneratedIdea]             # Combined insights
    execution_time: Optional[float] = None             # Total execution time
    metadata: Optional[Dict[str, Any]] = None          # Additional metadata
    
    def get_ideas_by_phase(self, phase: str) -> List[GeneratedIdea]:
        """Get ideas from specific phase."""
        
    def get_all_ideas(self) -> List[GeneratedIdea]:
        """Get all ideas from all phases."""
        
    def get_phase_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics for each phase."""
```

## Registry System

### Agent Registration

```python
from mad_spark_alt.core import register_agent, agent_registry
from mad_spark_alt.agents import QuestioningAgent

# Register individual agent
register_agent(QuestioningAgent)

# Check registration
agents = agent_registry.list_agents()
print(f"Registered agents: {list(agents.keys())}")

# Get agent by method
questioning_agent = agent_registry.get_agent_by_method(ThinkingMethod.QUESTIONING)

# Get agent by name
agent = agent_registry.get_agent("QuestioningAgent")
```

### Custom Agent Creation

```python
from mad_spark_alt.core import ThinkingAgentInterface, ThinkingMethod, register_agent
from mad_spark_alt.core.interfaces import IdeaGenerationRequest, IdeaGenerationResult, GeneratedIdea

class CustomThinkingAgent(ThinkingAgentInterface):
    """Custom thinking agent implementation."""
    
    @property
    def name(self) -> str:
        return "CustomThinkingAgent"
    
    @property
    def thinking_method(self) -> ThinkingMethod:
        # Use existing method or extend enum
        return ThinkingMethod.QUESTIONING
    
    @property
    def supported_output_types(self) -> List[OutputType]:
        return [OutputType.TEXT, OutputType.STRUCTURED]
    
    async def generate_ideas(self, request: IdeaGenerationRequest) -> IdeaGenerationResult:
        # Custom idea generation logic
        ideas = []
        
        for i in range(min(request.max_ideas_per_method, 3)):
            idea = GeneratedIdea(
                content=f"Custom idea {i+1}: {request.problem_statement}",
                thinking_method=self.thinking_method,
                agent_name=self.name,
                generation_prompt=f"Custom prompt for {request.problem_statement}",
                confidence_score=0.8,
                reasoning="Custom reasoning approach applied"
            )
            ideas.append(idea)
        
        return IdeaGenerationResult(
            agent_name=self.name,
            thinking_method=self.thinking_method,
            generated_ideas=ideas,
            execution_time=0.1
        )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        # Custom validation logic
        return True

# Register custom agent
register_agent(CustomThinkingAgent)
```

## Error Handling

### Robust Error Management

The QADI system includes comprehensive error handling:

```python
async def error_handling_example():
    # Orchestrator handles missing agents gracefully
    orchestrator = QADIOrchestrator()  # No agents registered
    
    result = await orchestrator.run_qadi_cycle(
        problem_statement="Test with missing agents",
        context="Testing error handling"
    )
    
    # Check for errors in each phase
    for phase_name, phase_result in result.phases.items():
        if phase_result.error_message:
            print(f"Phase {phase_name}: {phase_result.error_message}")
        else:
            print(f"Phase {phase_name}: Success")
```

### Error Types

| Error Type | Description | Handling |
|------------|-------------|----------|
| **Missing Agent** | No agent available for thinking method | Returns error result, continues cycle |
| **Generation Failure** | Agent fails during idea generation | Logs error, returns empty result |
| **Invalid Configuration** | Invalid agent configuration | Validation error, suggests corrections |
| **Timeout** | Agent takes too long to respond | Configurable timeout with fallback |

## Performance Considerations

### Optimization Strategies

1. **Parallel Processing**: Use `run_parallel_generation()` for independent operations
2. **Agent Caching**: Agents are cached after first instantiation
3. **Context Management**: Enhanced context is built incrementally
4. **Error Recovery**: Failed phases don't stop the entire cycle

### Monitoring and Metrics

```python
# Monitor execution times
result = await orchestrator.run_qadi_cycle(...)
print(f"Total time: {result.execution_time:.2f}s")

for phase_name, phase_result in result.phases.items():
    if phase_result.execution_time:
        print(f"{phase_name}: {phase_result.execution_time:.2f}s")

# Monitor idea generation rates
total_ideas = len(result.synthesized_ideas)
ideas_per_second = total_ideas / result.execution_time
print(f"Generation rate: {ideas_per_second:.2f} ideas/second")
```

## Integration with Evaluation System

The QADI system integrates seamlessly with the creativity evaluation system:

```python
from mad_spark_alt import CreativityEvaluator, EvaluationRequest, ModelOutput

async def evaluate_qadi_results():
    # Run QADI cycle
    orchestrator = QADIOrchestrator(agents)
    qadi_result = await orchestrator.run_qadi_cycle("Innovation challenge")
    
    # Convert ideas to evaluation format
    outputs = []
    for idea in qadi_result.synthesized_ideas:
        output = ModelOutput(
            content=idea.content,
            output_type=OutputType.TEXT,
            model_name=idea.agent_name,
            prompt=idea.generation_prompt
        )
        outputs.append(output)
    
    # Evaluate creativity
    evaluator = CreativityEvaluator()
    eval_summary = await evaluator.evaluate(EvaluationRequest(outputs=outputs))
    
    print(f"Overall creativity score: {eval_summary.get_overall_creativity_score():.3f}")
```

## Best Practices

### Configuration Management

```python
# Standard configuration template
qadi_config = {
    "max_ideas_per_method": 3,
    "require_reasoning": True,
    "creativity_level": "balanced",
    "thinking_depth": "thorough",
    "context_enhancement": True
}

# Agent-specific configurations
agent_configs = {
    "questioning": {
        "question_types": ["clarifying", "alternative"],
        "focus_areas": ["assumptions", "constraints"]
    },
    "abduction": {
        "hypothesis_types": ["causal", "analogical"],
        "creativity_level": "balanced"
    },
    "deduction": {
        "reasoning_modes": ["logical", "constraint"],
        "validation_depth": "thorough"
    },
    "induction": {
        "synthesis_methods": ["pattern", "principle"],
        "pattern_depth": "deep"
    }
}
```

### Testing Strategies

```python
import pytest
from mad_spark_alt.core import IdeaGenerationRequest
from mad_spark_alt.agents import QuestioningAgent

@pytest.mark.asyncio
async def test_questioning_agent():
    agent = QuestioningAgent()
    
    request = IdeaGenerationRequest(
        problem_statement="Test problem",
        max_ideas_per_method=2,
        require_reasoning=True
    )
    
    result = await agent.generate_ideas(request)
    
    assert result.agent_name == "QuestioningAgent"
    assert len(result.generated_ideas) <= 2
    assert all(idea.reasoning for idea in result.generated_ideas)
    assert result.error_message is None
```

This comprehensive API documentation covers all aspects of the QADI system, providing developers with the information needed to effectively use and extend the multi-agent idea generation framework.