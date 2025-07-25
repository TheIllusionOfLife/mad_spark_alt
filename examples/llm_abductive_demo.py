#!/usr/bin/env python3
"""
Demonstration of LLM-Powered Abductive Agent.

This script shows how the LLM Abductive Agent generates sophisticated hypotheses
using AI reasoning across multiple abductive strategies.

Usage:
    python examples/llm_abductive_demo.py
"""

import asyncio
import os
from typing import Optional, Tuple

from mad_spark_alt.agents.abduction.agent import AbductionAgent
from mad_spark_alt.agents.abduction.llm_agent import LLMAbductiveAgent
from mad_spark_alt.core.interfaces import IdeaGenerationRequest
from mad_spark_alt.core.llm_provider import LLMProvider, setup_llm_providers

# Constants
DEFAULT_MAX_IDEAS = 5
DEMO_SECTION_WIDTH = 60


async def setup_llm_agent() -> Optional[Tuple[LLMAbductiveAgent, str]]:
    """
    Setup LLM agent with API key checking and provider configuration.

    Returns:
        Tuple of (agent, status_message) if successful, None if no API keys available
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not (openai_key or anthropic_key):
        return None

    try:
        # Setup LLM providers
        await setup_llm_providers(
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
        )

        # Create the agent with preferred provider
        preferred_provider = LLMProvider.OPENAI if openai_key else LLMProvider.ANTHROPIC
        agent = LLMAbductiveAgent(preferred_provider=preferred_provider)

        provider_name = "OpenAI" if openai_key else "Anthropic"
        return agent, f"✅ LLM agent configured with {provider_name}"

    except (ValueError, RuntimeError, ConnectionError) as e:
        print(f"❌ Failed to setup LLM providers: {e}")
        return None


async def demo_basic_hypothesis_generation():
    """Demonstrate basic hypothesis generation with the LLM Abductive Agent."""
    print("🧠 LLM Abductive Agent - Basic Hypothesis Generation Demo")
    print("=" * DEMO_SECTION_WIDTH)

    # Setup LLM agent
    agent_setup = await setup_llm_agent()
    if not agent_setup:
        print("⚠️  No LLM API keys found in environment variables.")
        print(
            "   Please set OPENAI_API_KEY and/or ANTHROPIC_API_KEY to see AI-powered generation."
        )
        print("   Skipping LLM demonstration...")
        return

    agent, status_msg = agent_setup
    print(status_msg)

    # Test problem for hypothesis generation
    problem_statement = """
    Urban air quality has been steadily declining in major cities worldwide, 
    despite various emission reduction policies and technological advances. 
    Traditional approaches focusing on vehicle emissions and industrial regulation 
    haven't produced the expected improvements.
    """

    context = """
    This problem affects millions of urban residents and involves complex 
    interactions between transportation, industry, meteorology, and urban planning. 
    Recent data shows unexpected pollution patterns that don't align with 
    conventional models.
    """

    print(f"Problem: {problem_statement.strip()}")
    print(f"Context: {context.strip()}")
    print("\n🔍 Generating hypotheses using AI abductive reasoning...")

    # Create generation request
    request = IdeaGenerationRequest(
        problem_statement=problem_statement,
        context=context,
        max_ideas_per_method=6,
        generation_config={
            "max_strategies": 4,  # Only config key actually processed by agent
        },
    )

    try:
        # Generate hypotheses
        result = await agent.generate_ideas(request)

        print(
            f"\n✅ Generated {len(result.generated_ideas)} hypotheses in {result.execution_time:.2f}s"
        )
        print(
            f"📊 Strategies used: {', '.join(result.generation_metadata.get('strategies_used', []))}"
        )

        # Display each hypothesis
        for i, hypothesis in enumerate(result.generated_ideas, 1):
            print(f"\n{'='*50}")
            print(f"🔬 Hypothesis #{i}")
            print(f"Strategy: {hypothesis.metadata.get('strategy', 'unknown')}")
            print(f"Confidence: {hypothesis.confidence_score:.2f}")
            print(f"\n💡 {hypothesis.content}")
            print(f"\n🤔 Reasoning: {hypothesis.reasoning}")

            if hypothesis.metadata.get("implications"):
                print(f"\n📈 Implications: {hypothesis.metadata['implications']}")

            if hypothesis.metadata.get("evidence_requirements"):
                print(
                    f"\n🔍 Evidence Needed: {hypothesis.metadata['evidence_requirements']}"
                )

        # Show cost information
        total_cost = sum(h.metadata.get("llm_cost", 0) for h in result.generated_ideas)
        print(f"\n💰 Total LLM Cost: ${total_cost:.4f}")

    except Exception as e:
        print(f"❌ Error during hypothesis generation: {e}")


async def demo_strategy_comparison():
    """Demonstrate different abductive strategies in action."""
    print("\n\n🎯 Strategy Comparison Demo")
    print("=" * DEMO_SECTION_WIDTH)

    # Setup LLM agent using helper function
    agent_setup = await setup_llm_agent()
    if not agent_setup:
        print("⚠️  Skipping strategy comparison demo - no API keys available.")
        return

    agent, status_msg = agent_setup
    print(status_msg)

    # Different problem contexts to test strategy selection
    test_cases = [
        {
            "name": "High Complexity Problem",
            "problem": "Why do some software teams consistently deliver high-quality products while others struggle with the same tools and processes?",
            "config": {"max_strategies": 3},
            "expected_strategies": ["systems_perspective", "pattern_recognition"],
        },
        {
            "name": "Sparse Evidence Problem",
            "problem": "A new disease with unusual symptoms has emerged, but traditional diagnostic approaches are ineffective.",
            "config": {"max_strategies": 3},
            "expected_strategies": ["analogical_reasoning", "what_if_scenarios"],
        },
        {
            "name": "Counter-Intuitive Problem",
            "problem": "Employee productivity increased significantly during the transition to remote work, contrary to management expectations.",
            "config": {"max_strategies": 2},
            "expected_strategies": ["counter_intuitive"],
        },
    ]

    for test_case in test_cases:
        print(f"\n🔬 Testing: {test_case['name']}")
        print(f"Problem: {test_case['problem']}")

        request = IdeaGenerationRequest(
            problem_statement=test_case["problem"],
            max_ideas_per_method=4,
            generation_config=test_case["config"],
        )

        try:
            result = await agent.generate_ideas(request)
            strategies_used = result.generation_metadata.get("strategies_used", [])

            print(f"✅ Strategies selected: {', '.join(strategies_used)}")
            print(f"📝 Generated {len(result.generated_ideas)} hypotheses")

            # Show one example hypothesis
            if result.generated_ideas:
                example = result.generated_ideas[0]
                print(f"💡 Example: {example.content[:100]}...")

        except Exception as e:
            print(f"❌ Error: {e}")


async def demo_template_vs_llm_comparison():
    """Compare template-based vs LLM-powered hypothesis generation."""
    print("\n\n⚡ Template vs LLM Comparison")
    print("=" * DEMO_SECTION_WIDTH)

    # Test with same problem
    problem = "Traditional education systems are failing to prepare students for rapidly changing job markets."

    # Template-based agent
    template_agent = AbductionAgent()
    print("📝 Template-based Abductive Agent:")

    request = IdeaGenerationRequest(
        problem_statement=problem,
        max_ideas_per_method=DEFAULT_MAX_IDEAS,
    )

    template_result = await template_agent.generate_ideas(request)

    for i, idea in enumerate(template_result.generated_ideas, 1):
        print(f"  {i}. {idea.content}")

    # LLM-powered agent (if available)
    agent_setup = await setup_llm_agent()
    if agent_setup:
        agent, _ = agent_setup

        print(f"\n🧠 LLM-powered Abductive Agent:")

        try:
            llm_result = await agent.generate_ideas(request)

            for i, idea in enumerate(llm_result.generated_ideas, 1):
                strategy = idea.metadata.get("strategy", "unknown")
                print(f"  {i}. [{strategy}] {idea.content}")

            print(f"\n📊 Comparison:")
            print(
                f"  Template Agent: {len(template_result.generated_ideas)} hypotheses in {template_result.execution_time:.3f}s"
            )
            print(
                f"  LLM Agent: {len(llm_result.generated_ideas)} hypotheses in {llm_result.execution_time:.3f}s"
            )

            if llm_result.generated_ideas:
                total_cost = sum(
                    h.metadata.get("llm_cost", 0) for h in llm_result.generated_ideas
                )
                print(f"  LLM Cost: ${total_cost:.4f}")

        except Exception as e:
            print(f"❌ LLM agent failed: {e}")
    else:
        print("\n🧠 LLM-powered agent skipped (no API keys)")


async def main():
    """Run all demonstration scenarios."""
    print("🚀 LLM Abductive Agent Demonstration")
    print("=" * DEMO_SECTION_WIDTH)
    print("This demo shows how the LLM Abductive Agent generates sophisticated")
    print("hypotheses using AI reasoning across multiple abductive strategies.")
    print()

    try:
        await demo_basic_hypothesis_generation()
        await demo_strategy_comparison()
        await demo_template_vs_llm_comparison()

        print("\n🎉 Demo completed successfully!")
        print("\nThe LLM Abductive Agent demonstrates:")
        print("• Context-aware hypothesis generation")
        print("• Multiple abductive reasoning strategies")
        print("• Intelligent strategy selection")
        print("• Cost-effective AI integration")

    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
