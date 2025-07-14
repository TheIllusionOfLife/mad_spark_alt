# QADI Usage Examples

## Overview

This document provides comprehensive examples of using the Mad Spark Alt QADI system for various use cases, from basic idea generation to complex multi-phase thinking cycles.

## Dynamic Prompt Engineering Examples

### Automatic Question Type Detection

```python
from mad_spark_alt.core.prompt_classifier import classify_question

# Test different question types
questions = [
    "How to build a microservices architecture?",          # Technical
    "How can a startup compete with big tech?",            # Business
    "Design an interactive art installation",              # Creative
    "What factors influence remote work effectiveness?",    # Research
    "How to plan a product roadmap?",                      # Planning
    "How can I improve my daily productivity?"             # Personal
]

for question in questions:
    result = classify_question(question)
    print(f"Q: {question}")
    print(f"   Type: {result.question_type.value.title()}")
    print(f"   Confidence: {result.confidence:.1%}")
    print(f"   Complexity: {result.complexity.value.title()}")
    print()
```

### Adaptive Prompt Generation

```python
from mad_spark_alt.core.prompt_classifier import classify_question
from mad_spark_alt.core.adaptive_prompts import get_adaptive_prompt

# Classify and generate adaptive prompts
question = "How to build a robust microservices architecture?"
classification = classify_question(question)

# Get different types of prompts
regular_prompt = get_adaptive_prompt(
    phase_name="questioning",
    classification_result=classification,
    prompt=question,
    concrete_mode=False
)

concrete_prompt = get_adaptive_prompt(
    phase_name="questioning", 
    classification_result=classification,
    prompt=question,
    concrete_mode=True
)

print(f"Regular mode prompt:\n{regular_prompt}\n")
print(f"Concrete mode prompt:\n{concrete_prompt}")
```

### CLI Usage with Prompt Engineering

```bash
# Let the system auto-detect question type
uv run python qadi_simple_multi.py "How to scale a microservices architecture?"

# Override to force business perspective on technical question
uv run python qadi_simple_multi.py --type=business "How to scale microservices?"

# Use concrete mode for implementation-focused analysis
uv run python qadi_simple_multi.py --concrete "Build a microservices system"

# Combine type override with concrete mode
uv run python qadi_simple_multi.py --type=technical --concrete "Design REST APIs"
```

## Quick Start Examples

### Basic Agent Usage

```python
import asyncio
from mad_spark_alt.agents import QuestioningAgent
from mad_spark_alt.core import IdeaGenerationRequest

async def basic_questioning_example():
    """Basic usage of the QuestioningAgent."""
    agent = QuestioningAgent()
    
    request = IdeaGenerationRequest(
        problem_statement="How can we reduce plastic waste in urban environments?",
        context="City with 2 million population, focusing on practical solutions",
        max_ideas_per_method=3,
        require_reasoning=True
    )
    
    result = await agent.generate_ideas(request)
    
    print(f"Agent: {result.agent_name}")
    print(f"Generated {len(result.generated_ideas)} questions:")
    
    for i, idea in enumerate(result.generated_ideas, 1):
        print(f"\n{i}. {idea.content}")
        if idea.reasoning:
            print(f"   💭 Reasoning: {idea.reasoning}")
        print(f"   🎯 Strategy: {idea.metadata.get('strategy', 'general')}")

# Run the example
asyncio.run(basic_questioning_example())
```

### Complete QADI Cycle

```python
import asyncio
from mad_spark_alt.core import QADIOrchestrator, agent_registry, register_agent
from mad_spark_alt.agents import QuestioningAgent, AbductionAgent, DeductionAgent, InductionAgent

async def complete_qadi_example():
    """Complete QADI cycle demonstration."""
    
    # Step 1: Register all thinking agents
    register_agent(QuestioningAgent)
    register_agent(AbductionAgent)
    register_agent(DeductionAgent)
    register_agent(InductionAgent)
    
    # Step 2: Create agent list
    agents = [
        agent_registry.get_agent_by_method(method) 
        for method in [ThinkingMethod.QUESTIONING, ThinkingMethod.ABDUCTION, 
                      ThinkingMethod.DEDUCTION, ThinkingMethod.INDUCTION]
    ]
    
    # Step 3: Initialize orchestrator
    orchestrator = QADIOrchestrator([a for a in agents if a])
    
    # Step 4: Run QADI cycle
    result = await orchestrator.run_qadi_cycle(
        problem_statement="How can we make remote work more effective and engaging?",
        context="Post-pandemic workplace, distributed teams, technology-enabled collaboration",
        cycle_config={
            "max_ideas_per_method": 3,
            "require_reasoning": True,
            "creativity_level": "balanced"
        }
    )
    
    # Step 5: Display results
    print(f"🔄 QADI Cycle completed in {result.execution_time:.2f}s")
    print(f"🆔 Cycle ID: {result.cycle_id}")
    
    phase_emojis = {
        "questioning": "❓",
        "abduction": "💡", 
        "deduction": "🔍",
        "induction": "🔗"
    }
    
    for phase_name, phase_result in result.phases.items():
        emoji = phase_emojis.get(phase_name, "🧠")
        print(f"\n{emoji} {phase_name.title()} Phase:")
        print(f"   Agent: {phase_result.agent_name}")
        
        if phase_result.error_message:
            print(f"   ❌ Error: {phase_result.error_message}")
        else:
            for i, idea in enumerate(phase_result.generated_ideas, 1):
                print(f"   {i}. {idea.content}")
                if idea.reasoning and len(idea.reasoning) < 100:
                    print(f"      💭 {idea.reasoning}")
    
    print(f"\n🎨 Total synthesized ideas: {len(result.synthesized_ideas)}")

# Run the example
asyncio.run(complete_qadi_example())
```

## Domain-Specific Examples

### Business Innovation

```python
import asyncio
from mad_spark_alt.core import QADIOrchestrator, ThinkingMethod

async def business_innovation_example():
    """QADI cycle for business innovation challenges."""
    
    orchestrator = QADIOrchestrator()
    
    # Business innovation problem
    result = await orchestrator.run_qadi_cycle(
        problem_statement="How can we increase customer retention in our SaaS business?",
        context="B2B SaaS platform, 500+ customers, monthly churn rate 5%, competitive market",
        cycle_config={
            "max_ideas_per_method": 4,
            "require_reasoning": True,
            "creativity_level": "balanced"
        }
    )
    
    # Analyze business-specific insights
    business_insights = []
    for phase_name, phase_result in result.phases.items():
        for idea in phase_result.generated_ideas:
            if any(keyword in idea.content.lower() for keyword in 
                   ['customer', 'retention', 'churn', 'value', 'engagement']):
                business_insights.append({
                    'phase': phase_name,
                    'insight': idea.content,
                    'confidence': idea.confidence_score
                })
    
    print("📈 Business Innovation Insights:")
    for insight in sorted(business_insights, key=lambda x: x['confidence'], reverse=True):
        print(f"   {insight['phase'].title()}: {insight['insight']}")
        print(f"   Confidence: {insight['confidence']:.2f}\n")

asyncio.run(business_innovation_example())
```

### Educational Technology

```python
async def education_technology_example():
    """QADI cycle for educational technology challenges."""
    
    orchestrator = QADIOrchestrator()
    
    result = await orchestrator.run_qadi_cycle(
        problem_statement="How can we improve student engagement in online learning?",
        context="University-level courses, remote learning environment, diverse student backgrounds",
        cycle_config={
            "max_ideas_per_method": 3,
            "require_reasoning": True
        }
    )
    
    # Extract educational strategies
    educational_strategies = {}
    for phase_name, phase_result in result.phases.items():
        strategies = []
        for idea in phase_result.generated_ideas:
            if any(keyword in idea.content.lower() for keyword in 
                   ['student', 'learning', 'engagement', 'interactive', 'motivation']):
                strategies.append(idea.content)
        educational_strategies[phase_name] = strategies
    
    print("🎓 Educational Technology Strategies:")
    for phase, strategies in educational_strategies.items():
        if strategies:
            print(f"\n{phase.title()} Strategies:")
            for i, strategy in enumerate(strategies, 1):
                print(f"   {i}. {strategy}")

asyncio.run(education_technology_example())
```

### Scientific Research

```python
async def scientific_research_example():
    """QADI cycle for scientific research questions."""
    
    # Focus on questioning and abduction for research
    orchestrator = QADIOrchestrator()
    
    # Generate research questions and hypotheses
    parallel_results = await orchestrator.run_parallel_generation(
        problem_statement="How can artificial intelligence accelerate drug discovery?",
        thinking_methods=[ThinkingMethod.QUESTIONING, ThinkingMethod.ABDUCTION],
        context="Pharmaceutical research, computational biology, machine learning applications",
        config={
            "max_ideas_per_method": 5,
            "require_reasoning": True
        }
    )
    
    print("🔬 Scientific Research Generation:")
    
    # Process questioning results
    if ThinkingMethod.QUESTIONING in parallel_results:
        questions = parallel_results[ThinkingMethod.QUESTIONING]
        print(f"\n❓ Research Questions ({len(questions.generated_ideas)} generated):")
        for i, idea in enumerate(questions.generated_ideas, 1):
            print(f"   {i}. {idea.content}")
    
    # Process abduction results  
    if ThinkingMethod.ABDUCTION in parallel_results:
        hypotheses = parallel_results[ThinkingMethod.ABDUCTION]
        print(f"\n💡 Research Hypotheses ({len(hypotheses.generated_ideas)} generated):")
        for i, idea in enumerate(hypotheses.generated_ideas, 1):
            print(f"   {i}. {idea.content}")
            print(f"      Confidence: {idea.confidence_score:.2f}")

asyncio.run(scientific_research_example())
```

## Advanced Usage Patterns

### Custom Agent Configuration

```python
async def custom_configuration_example():
    """Example with custom agent configurations."""
    
    from mad_spark_alt.agents import QuestioningAgent, AbductionAgent
    
    # Custom questioning configuration
    questioning_agent = QuestioningAgent()
    questioning_request = IdeaGenerationRequest(
        problem_statement="How can we create more sustainable urban transportation?",
        context="Large metropolitan area, environmental concerns, budget constraints",
        max_ideas_per_method=4,
        require_reasoning=True,
        generation_config={
            "question_types": ["clarifying", "challenging", "expanding"],
            "focus_areas": ["stakeholders", "constraints", "alternatives"]
        }
    )
    
    questioning_result = await questioning_agent.generate_ideas(questioning_request)
    
    # Custom abduction configuration
    abduction_agent = AbductionAgent()
    abduction_request = IdeaGenerationRequest(
        problem_statement="How can we create more sustainable urban transportation?",
        context="Building on previous questions: " + 
                "; ".join([idea.content for idea in questioning_result.generated_ideas]),
        max_ideas_per_method=3,
        require_reasoning=True,
        generation_config={
            "hypothesis_types": ["analogical", "emergent"],
            "creativity_level": "high",
            "use_analogies": True
        }
    )
    
    abduction_result = await abduction_agent.generate_ideas(abduction_request)
    
    print("🚀 Custom Configuration Results:")
    print(f"\n❓ Strategic Questions:")
    for idea in questioning_result.generated_ideas:
        focus = idea.metadata.get('focus_area', 'general')
        print(f"   [{focus}] {idea.content}")
    
    print(f"\n💡 Creative Hypotheses:")
    for idea in abduction_result.generated_ideas:
        strategy = idea.metadata.get('strategy', 'creative_leap')
        print(f"   [{strategy}] {idea.content}")

asyncio.run(custom_configuration_example())
```

### Iterative Idea Development

```python
async def iterative_development_example():
    """Example showing iterative idea development."""
    
    orchestrator = QADIOrchestrator()
    
    # Initial problem
    initial_problem = "How can we reduce food waste in restaurants?"
    
    # Round 1: Initial QADI cycle
    print("🔄 Round 1: Initial Analysis")
    round1_result = await orchestrator.run_qadi_cycle(
        problem_statement=initial_problem,
        context="Restaurant industry, sustainability focus, cost considerations",
        cycle_config={"max_ideas_per_method": 2}
    )
    
    # Extract key insights from round 1
    key_insights = []
    for phase_result in round1_result.phases.values():
        for idea in phase_result.generated_ideas:
            if idea.confidence_score > 0.7:
                key_insights.append(idea.content)
    
    # Round 2: Refined analysis based on insights
    print("\n🔄 Round 2: Refined Analysis")
    refined_context = f"Building on insights: {'; '.join(key_insights[:3])}"
    
    round2_result = await orchestrator.run_qadi_cycle(
        problem_statement="How can we implement the most promising food waste reduction strategies?",
        context=refined_context,
        cycle_config={"max_ideas_per_method": 3, "require_reasoning": True}
    )
    
    print(f"\n📊 Iteration Results:")
    print(f"Round 1 insights: {len(round1_result.synthesized_ideas)}")
    print(f"Round 2 refinements: {len(round2_result.synthesized_ideas)}")
    
    # Show evolution of thinking
    print(f"\n🎯 Final Refined Ideas:")
    for i, idea in enumerate(round2_result.synthesized_ideas[:5], 1):
        phase = idea.metadata.get('phase', 'unknown')
        print(f"   {i}. [{phase}] {idea.content}")

asyncio.run(iterative_development_example())
```

### Integration with Evaluation System

```python
from mad_spark_alt import CreativityEvaluator, EvaluationRequest, ModelOutput, OutputType

async def evaluation_integration_example():
    """Example integrating QADI generation with creativity evaluation."""
    
    # Step 1: Generate ideas using QADI
    orchestrator = QADIOrchestrator()
    
    qadi_result = await orchestrator.run_qadi_cycle(
        problem_statement="Design an innovative mobile app feature",
        context="Social media platform, Gen-Z users, privacy-conscious",
        cycle_config={"max_ideas_per_method": 3}
    )
    
    # Step 2: Convert ideas to evaluation format
    outputs = []
    for idea in qadi_result.synthesized_ideas:
        output = ModelOutput(
            content=idea.content,
            output_type=OutputType.TEXT,
            model_name=idea.agent_name,
            prompt=idea.generation_prompt
        )
        outputs.append(output)
    
    # Step 3: Evaluate creativity
    evaluator = CreativityEvaluator()
    eval_summary = await evaluator.evaluate(EvaluationRequest(outputs=outputs))
    
    # Step 4: Analyze results
    print("🎨 QADI + Evaluation Integration:")
    print(f"Generated ideas: {len(outputs)}")
    print(f"Overall creativity score: {eval_summary.get_overall_creativity_score():.3f}")
    
    # Show top-rated ideas
    idea_scores = []
    for output, layer_results in zip(outputs, eval_summary.layer_results.get('quantitative', [])):
        for result in layer_results:
            if result.evaluator_name == 'diversity_evaluator':
                novelty = result.scores.get('novelty_score', 0)
                idea_scores.append((output.content, novelty))
    
    # Sort by novelty score
    idea_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 Top Creative Ideas (by novelty):")
    for i, (content, score) in enumerate(idea_scores[:3], 1):
        print(f"   {i}. [Score: {score:.3f}] {content}")

asyncio.run(evaluation_integration_example())
```

## Error Handling Examples

### Robust Error Handling

```python
async def error_handling_example():
    """Example showing robust error handling."""
    
    from mad_spark_alt.core import agent_registry
    
    # Clear registry to simulate missing agents
    agent_registry.clear()
    
    orchestrator = QADIOrchestrator()
    
    try:
        result = await orchestrator.run_qadi_cycle(
            problem_statement="Test problem with missing agents",
            context="Testing error resilience"
        )
        
        print("🛡️ Error Handling Demonstration:")
        print(f"Cycle completed despite missing agents")
        print(f"Execution time: {result.execution_time:.2f}s")
        
        # Check which phases had errors
        error_phases = []
        success_phases = []
        
        for phase_name, phase_result in result.phases.items():
            if phase_result.error_message:
                error_phases.append(phase_name)
                print(f"❌ {phase_name}: {phase_result.error_message}")
            else:
                success_phases.append(phase_name)
                print(f"✅ {phase_name}: {len(phase_result.generated_ideas)} ideas")
        
        print(f"\nSummary: {len(error_phases)} errors, {len(success_phases)} successes")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Re-register agents for subsequent examples
    from mad_spark_alt.agents import QuestioningAgent, AbductionAgent, DeductionAgent, InductionAgent
    register_agent(QuestioningAgent)
    register_agent(AbductionAgent)  
    register_agent(DeductionAgent)
    register_agent(InductionAgent)

asyncio.run(error_handling_example())
```

### Graceful Degradation

```python
async def graceful_degradation_example():
    """Example showing graceful degradation with partial agent availability."""
    
    from mad_spark_alt.core import agent_registry
    from mad_spark_alt.agents import QuestioningAgent, AbductionAgent
    
    # Register only some agents
    agent_registry.clear()
    register_agent(QuestioningAgent)
    register_agent(AbductionAgent)
    
    orchestrator = QADIOrchestrator()
    
    result = await orchestrator.run_qadi_cycle(
        problem_statement="Innovation challenge with limited agents",
        context="Testing partial functionality"
    )
    
    print("🔄 Graceful Degradation Example:")
    
    available_phases = []
    missing_phases = []
    
    for phase_name, phase_result in result.phases.items():
        if phase_result.error_message:
            missing_phases.append(phase_name)
        else:
            available_phases.append(phase_name)
            print(f"✅ {phase_name}: Generated {len(phase_result.generated_ideas)} ideas")
    
    print(f"\n📊 Results with partial agents:")
    print(f"Available phases: {', '.join(available_phases)}")
    print(f"Missing phases: {', '.join(missing_phases)}")
    print(f"Still generated {len(result.synthesized_ideas)} total ideas")

asyncio.run(graceful_degradation_example())
```

## Performance Examples

### Parallel Processing

```python
async def parallel_processing_example():
    """Example demonstrating parallel processing capabilities."""
    
    import time
    
    orchestrator = QADIOrchestrator()
    
    # Sequential processing
    start_time = time.time()
    
    sequential_results = []
    for method in [ThinkingMethod.QUESTIONING, ThinkingMethod.ABDUCTION]:
        result = await orchestrator.run_parallel_generation(
            problem_statement="Optimize team productivity",
            thinking_methods=[method],
            context="Remote work environment",
            config={"max_ideas_per_method": 2}
        )
        sequential_results.append(result)
    
    sequential_time = time.time() - start_time
    
    # Parallel processing
    start_time = time.time()
    
    parallel_result = await orchestrator.run_parallel_generation(
        problem_statement="Optimize team productivity", 
        thinking_methods=[ThinkingMethod.QUESTIONING, ThinkingMethod.ABDUCTION],
        context="Remote work environment",
        config={"max_ideas_per_method": 2}
    )
    
    parallel_time = time.time() - start_time
    
    print("⚡ Performance Comparison:")
    print(f"Sequential processing: {sequential_time:.2f}s")
    print(f"Parallel processing: {parallel_time:.2f}s")
    print(f"Speedup: {sequential_time/parallel_time:.2f}x")
    
    # Show parallel results
    for method, result in parallel_result.items():
        print(f"\n{method.value.title()} Results:")
        for idea in result.generated_ideas:
            print(f"   • {idea.content}")

asyncio.run(parallel_processing_example())
```

### Batch Processing

```python
async def batch_processing_example():
    """Example of processing multiple problems efficiently."""
    
    orchestrator = QADIOrchestrator()
    
    problems = [
        "How can we improve customer satisfaction?",
        "What are innovative ways to reduce costs?", 
        "How can we enhance team collaboration?",
        "What strategies can boost employee engagement?"
    ]
    
    print("📦 Batch Processing Example:")
    
    # Process multiple problems
    batch_results = []
    start_time = time.time()
    
    for i, problem in enumerate(problems, 1):
        print(f"Processing problem {i}/{len(problems)}...")
        
        result = await orchestrator.run_qadi_cycle(
            problem_statement=problem,
            context="Business optimization context",
            cycle_config={"max_ideas_per_method": 2}
        )
        
        batch_results.append({
            'problem': problem,
            'ideas_count': len(result.synthesized_ideas),
            'execution_time': result.execution_time,
            'top_idea': result.synthesized_ideas[0].content if result.synthesized_ideas else "No ideas generated"
        })
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n📊 Batch Results Summary:")
    print(f"Total problems processed: {len(problems)}")
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Average time per problem: {total_time/len(problems):.2f}s")
    
    total_ideas = sum(r['ideas_count'] for r in batch_results)
    print(f"Total ideas generated: {total_ideas}")
    
    print(f"\n🎯 Top Ideas from Each Problem:")
    for i, result in enumerate(batch_results, 1):
        print(f"   {i}. {result['top_idea']}")

asyncio.run(batch_processing_example())
```

## Testing and Validation Examples

### System Testing

```python
async def system_testing_example():
    """Example of comprehensive system testing."""
    
    print("🧪 System Testing Example:")
    
    # Test 1: Agent registration
    print("\n1. Testing agent registration...")
    from mad_spark_alt.core import agent_registry
    
    initial_count = len(agent_registry.list_agents())
    print(f"   Initial agents: {initial_count}")
    
    # Test 2: Individual agent functionality
    print("\n2. Testing individual agents...")
    from mad_spark_alt.agents import QuestioningAgent
    
    agent = QuestioningAgent()
    test_request = IdeaGenerationRequest(
        problem_statement="Test problem",
        max_ideas_per_method=1
    )
    
    test_result = await agent.generate_ideas(test_request)
    assert test_result.agent_name == "QuestioningAgent"
    assert len(test_result.generated_ideas) <= 1
    print(f"   ✅ QuestioningAgent: Generated {len(test_result.generated_ideas)} ideas")
    
    # Test 3: Orchestrator functionality
    print("\n3. Testing orchestrator...")
    orchestrator = QADIOrchestrator()
    
    cycle_result = await orchestrator.run_qadi_cycle(
        problem_statement="Test orchestrator functionality",
        cycle_config={"max_ideas_per_method": 1}
    )
    
    assert cycle_result.cycle_id is not None
    assert cycle_result.execution_time is not None
    print(f"   ✅ Orchestrator: Completed cycle in {cycle_result.execution_time:.2f}s")
    
    # Test 4: Error resilience
    print("\n4. Testing error resilience...")
    
    # Test with empty problem statement
    try:
        empty_result = await orchestrator.run_qadi_cycle(
            problem_statement="",
            cycle_config={"max_ideas_per_method": 1}
        )
        print(f"   ✅ Error resilience: Handled empty input gracefully")
    except Exception as e:
        print(f"   ⚠️ Error resilience: Exception occurred: {e}")
    
    print(f"\n🎉 System testing completed successfully!")

asyncio.run(system_testing_example())
```

### Custom Agent Testing

```python
async def custom_agent_testing():
    """Example of testing custom agent implementations."""
    
    from mad_spark_alt.core import ThinkingAgentInterface, ThinkingMethod, register_agent
    from mad_spark_alt.core.interfaces import GeneratedIdea
    
    class TestAgent(ThinkingAgentInterface):
        """Test agent for validation."""
        
        @property
        def name(self) -> str:
            return "TestAgent"
        
        @property  
        def thinking_method(self) -> ThinkingMethod:
            return ThinkingMethod.QUESTIONING
        
        @property
        def supported_output_types(self) -> List[OutputType]:
            return [OutputType.TEXT]
        
        async def generate_ideas(self, request: IdeaGenerationRequest) -> IdeaGenerationResult:
            ideas = []
            for i in range(min(request.max_ideas_per_method, 2)):
                idea = GeneratedIdea(
                    content=f"Test idea {i+1} for: {request.problem_statement}",
                    thinking_method=self.thinking_method,
                    agent_name=self.name,
                    generation_prompt="Test generation",
                    confidence_score=0.9
                )
                ideas.append(idea)
            
            return IdeaGenerationResult(
                agent_name=self.name,
                thinking_method=self.thinking_method,
                generated_ideas=ideas,
                execution_time=0.1
            )
        
        def validate_config(self, config: Dict[str, Any]) -> bool:
            return True
    
    print("🔧 Custom Agent Testing:")
    
    # Test agent creation
    test_agent = TestAgent()
    print(f"   ✅ Created agent: {test_agent.name}")
    
    # Test agent registration
    register_agent(TestAgent)
    print(f"   ✅ Registered agent successfully")
    
    # Test agent functionality
    test_request = IdeaGenerationRequest(
        problem_statement="Custom agent test",
        max_ideas_per_method=2
    )
    
    result = await test_agent.generate_ideas(test_request)
    print(f"   ✅ Generated {len(result.generated_ideas)} ideas")
    
    # Validate output format
    for idea in result.generated_ideas:
        assert idea.content is not None
        assert idea.agent_name == "TestAgent"
        assert idea.confidence_score >= 0 and idea.confidence_score <= 1
    
    print(f"   ✅ Output validation passed")
    print(f"   📝 Sample idea: {result.generated_ideas[0].content}")

asyncio.run(custom_agent_testing())
```

This comprehensive examples documentation provides developers and users with practical, ready-to-run code demonstrating the full capabilities of the QADI system across various domains and use cases.