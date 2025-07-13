#!/usr/bin/env python3
"""
Demonstrate custom agent configuration and registry usage.
"""

import asyncio
from mad_spark_alt.core import (
    QADIOrchestrator, 
    ThinkingAgentRegistry,
    IdeaGenerationRequest,
    ThinkingMethod
)
from mad_spark_alt.agents import QuestioningAgent, AbductionAgent

async def custom_configuration_demo():
    """Show how to customize agent behavior and configuration."""
    
    print("🛠️  Custom Configuration Demo")
    print("=" * 50)
    
    # Create custom registry
    custom_registry = ThinkingAgentRegistry()
    
    # Register only specific agents
    custom_registry.register(QuestioningAgent)
    custom_registry.register(AbductionAgent)
    
    print(f"📋 Custom registry has {len(custom_registry.get_all_agents())} agents")
    
    # Test individual agent with custom request
    questioning_agent = custom_registry.get_agent_by_method(ThinkingMethod.QUESTIONING)
    
    if questioning_agent:
        print(f"\n🤔 Testing {questioning_agent.name} individually...")
        
        custom_request = IdeaGenerationRequest(
            problem_statement="How can we make cities more bike-friendly?",
            context="Consider infrastructure, policy, and cultural changes",
            max_ideas_per_method=4,
            require_reasoning=True,
            generation_config={
                "creativity_level": "high",
                "detail_level": "comprehensive"
            }
        )
        
        result = await questioning_agent.generate_ideas(custom_request)
        
        print(f"✅ Generated {len(result.generated_ideas)} questions:")
        for i, idea in enumerate(result.generated_ideas, 1):
            print(f"  {i}. {idea.content}")
            if idea.reasoning:
                print(f"     💭 {idea.reasoning}")
    
    print("\n" + "=" * 50)
    
    # Create orchestrator with limited agents
    limited_agents = [QuestioningAgent(), AbductionAgent()]
    orchestrator = QADIOrchestrator(agents=limited_agents)
    
    print(f"🔧 Limited orchestrator with {len(orchestrator.agents)} agents")
    print("📝 Available methods:", list(orchestrator.agents.keys()))
    
    # Try to run cycle (will show missing agent handling)
    result = await orchestrator.run_qadi_cycle(
        problem_statement="How can we improve urban air quality?",
        cycle_config={"max_ideas_per_method": 2}
    )
    
    print(f"\n📊 Cycle Results:")
    print(f"   Execution time: {result.execution_time:.3f}s")
    print(f"   Phases completed: {len([p for p in result.phases.values() if p])}")
    print(f"   Total ideas: {len(result.synthesized_ideas)}")
    
    # Show which phases worked
    for phase_name, phase_result in result.phases.items():
        if phase_result and phase_result.generated_ideas:
            print(f"   ✅ {phase_name}: {len(phase_result.generated_ideas)} ideas")
        else:
            print(f"   ❌ {phase_name}: No agent available")

if __name__ == "__main__":
    asyncio.run(custom_configuration_demo())