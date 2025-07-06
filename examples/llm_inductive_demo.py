#!/usr/bin/env python3
"""
Demonstration of LLM-powered Inductive Agent capabilities.

This script showcases the intelligent pattern synthesis and insight generation
capabilities of the LLM Inductive Agent compared to template-based approaches.
"""

import asyncio
import logging
from typing import Dict, Any

from mad_spark_alt.agents.induction.agent import InductionAgent
from mad_spark_alt.agents.induction.llm_agent import LLMInductiveAgent
from mad_spark_alt.core.interfaces import IdeaGenerationRequest

# Configure logging for demo
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def print_agent_result(agent_name: str, result, show_metadata: bool = True):
    """Print results from an agent in a formatted way."""
    print(f"\n🤖 {agent_name} Results:")
    print(f"   Execution Time: {result.execution_time:.2f}s")
    print(f"   Ideas Generated: {len(result.generated_ideas)}")

    if result.error_message:
        print(f"   ❌ Error: {result.error_message}")
        return

    for i, idea in enumerate(result.generated_ideas, 1):
        print(f"\n   💡 Insight {i}:")
        print(f"      Content: {idea.content}")
        print(f"      Confidence: {idea.confidence_score:.2f}")
        print(f"      Reasoning: {idea.reasoning}")

        if show_metadata and hasattr(idea, "metadata") and idea.metadata:
            method = idea.metadata.get("method", "unknown")
            print(f"      Method: {method}")

            # Show LLM-specific metadata for LLM agents
            if "llm_cost" in idea.metadata:
                cost = idea.metadata.get("llm_cost", 0)
                print(f"      Cost: ${cost:.4f}")

            # Show synthesis-specific metadata
            if "supporting_patterns" in idea.metadata:
                patterns = idea.metadata.get("supporting_patterns", "none")
                print(f"      Supporting Patterns: {patterns}")

            if "generalization_scope" in idea.metadata:
                scope = idea.metadata.get("generalization_scope", "unknown")
                print(f"      Generalization Scope: {scope}")

    if show_metadata and hasattr(result, "generation_metadata"):
        metadata = result.generation_metadata
        if "synthesis_context" in metadata:
            context = metadata["synthesis_context"]
            print(f"\n   📊 Synthesis Context:")
            print(f"      Data Richness: {context.get('data_richness', 'unknown')}")
            print(
                f"      Pattern Visibility: {context.get('pattern_visibility', 'unknown')}"
            )
            print(
                f"      Generalization Potential: {context.get('generalization_potential', 'unknown')}"
            )


async def demo_basic_pattern_synthesis():
    """Demonstrate basic pattern synthesis capabilities."""
    print_section("Basic Pattern Synthesis Demonstration")

    # Test problem about software development patterns
    request = IdeaGenerationRequest(
        problem_statement="How can we identify and leverage recurring patterns in software development to improve code quality and team productivity?",
        context="Software engineering, development practices, team collaboration, code quality metrics",
        max_ideas_per_method=3,
    )

    print("🎯 Problem: Software Development Pattern Recognition")
    print(f"Statement: {request.problem_statement}")
    print(f"Context: {request.context}")

    # Compare template-based vs LLM-powered approaches
    template_agent = InductionAgent(name="TemplateInductiveAgent")
    llm_agent = LLMInductiveAgent(name="LLMInductiveAgent")

    print_subsection("Template-Based Inductive Agent")
    template_result = await template_agent.generate_ideas(request)
    print_agent_result("Template Inductive Agent", template_result, show_metadata=False)

    print_subsection("LLM-Powered Inductive Agent")
    llm_result = await llm_agent.generate_ideas(request)
    print_agent_result("LLM Inductive Agent", llm_result, show_metadata=True)


async def demo_scientific_discovery():
    """Demonstrate pattern synthesis for scientific discovery."""
    print_section("Scientific Discovery Pattern Synthesis")

    request = IdeaGenerationRequest(
        problem_statement="What patterns can we observe from recent breakthroughs in renewable energy technology that might predict future innovation directions?",
        context="Renewable energy, solar panels, wind turbines, battery technology, grid integration, efficiency improvements",
        max_ideas_per_method=4,
        generation_config={
            "inductive_method": "trend_analysis",
            "synthesis_scope": "broad",
            "include_emergent_properties": True,
        },
    )

    print("🎯 Problem: Renewable Energy Innovation Patterns")
    print(f"Statement: {request.problem_statement}")
    print(f"Context: {request.context}")

    llm_agent = LLMInductiveAgent(name="EnergyPatternAnalyst")

    print_subsection("LLM Pattern Synthesis Results")
    result = await llm_agent.generate_ideas(request)
    print_agent_result("Energy Pattern Analyst", result)


async def demo_business_intelligence():
    """Demonstrate pattern synthesis for business intelligence."""
    print_section("Business Intelligence Pattern Analysis")

    request = IdeaGenerationRequest(
        problem_statement="How can we synthesize patterns from customer behavior data to predict emerging market opportunities and business model innovations?",
        context="Customer analytics, market trends, business models, e-commerce, digital transformation, consumer psychology",
        max_ideas_per_method=3,
        generation_config={
            "inductive_method": "meta_recognition",
            "pattern_depth": "deep",
            "creative_synthesis": True,
        },
    )

    print("🎯 Problem: Customer Behavior Pattern Analysis")
    print(f"Statement: {request.problem_statement}")
    print(f"Context: {request.context}")

    llm_agent = LLMInductiveAgent(name="BusinessIntelligenceAgent")

    print_subsection("Business Pattern Insights")
    result = await llm_agent.generate_ideas(request)
    print_agent_result("Business Intelligence Agent", result)


async def demo_method_comparison():
    """Demonstrate different inductive methods on the same problem."""
    print_section("Inductive Method Comparison")

    base_problem = "How can we understand the patterns behind successful remote team collaboration?"
    base_context = "Remote work, team dynamics, communication tools, productivity, collaboration platforms"

    print("🎯 Problem: Remote Team Collaboration Patterns")
    print(f"Statement: {base_problem}")
    print(f"Context: {base_context}")

    # Test different inductive methods
    methods_to_test = [
        ("pattern_synthesis", "Pattern-focused synthesis"),
        ("principle_extraction", "Principle extraction approach"),
        ("creative_synthesis", "Creative combination method"),
        ("analogical_extension", "Cross-domain analogies"),
    ]

    for method_name, method_description in methods_to_test:
        print_subsection(f"{method_description}")

        request = IdeaGenerationRequest(
            problem_statement=base_problem,
            context=base_context,
            max_ideas_per_method=2,
            generation_config={
                "inductive_method": method_name,
                "max_methods": 1,  # Force single method
            },
        )

        llm_agent = LLMInductiveAgent(name=f"{method_name.title()}Agent")
        result = await llm_agent.generate_ideas(request)
        print_agent_result(f"{method_description} Agent", result)


async def demo_complex_synthesis():
    """Demonstrate complex multi-dimensional pattern synthesis."""
    print_section("Complex Multi-Dimensional Pattern Synthesis")

    request = IdeaGenerationRequest(
        problem_statement="What emerging patterns can we identify at the intersection of artificial intelligence, climate change mitigation, and social equity that suggest new approaches to sustainable development?",
        context="AI technology, climate science, social justice, sustainable development, environmental policy, technological ethics, global cooperation",
        max_ideas_per_method=3,
        generation_config={
            "synthesis_scope": "broad",
            "generalization_level": "high",
            "include_emergent_properties": True,
            "meta_pattern_analysis": True,
        },
    )

    print("🎯 Problem: AI-Climate-Equity Pattern Synthesis")
    print(f"Statement: {request.problem_statement}")
    print(f"Context: {request.context}")

    llm_agent = LLMInductiveAgent(name="ComplexSynthesisAgent")

    print_subsection("Multi-Dimensional Pattern Analysis")
    result = await llm_agent.generate_ideas(request)
    print_agent_result("Complex Synthesis Agent", result)


async def demo_error_handling():
    """Demonstrate error handling and fallback mechanisms."""
    print_section("Error Handling and Fallback Mechanisms")

    # This will work if LLM services are available
    request = IdeaGenerationRequest(
        problem_statement="How can we build resilient systems that gracefully handle unexpected failures?",
        context="System reliability, fault tolerance, graceful degradation",
        max_ideas_per_method=2,
    )

    print("🎯 Problem: Resilient System Patterns")
    print(f"Statement: {request.problem_statement}")

    llm_agent = LLMInductiveAgent(name="ResilienceAgent")

    print_subsection("Testing System Resilience")
    result = await llm_agent.generate_ideas(request)
    print_agent_result("Resilience Agent", result)

    if result.error_message:
        print(
            "\n💡 Note: This demonstrates graceful error handling when LLM services are unavailable."
        )
        print("   The agent returns structured error information rather than crashing.")
    else:
        print("\n✅ LLM services are available and working correctly.")


def print_cost_summary(results):
    """Print a summary of total costs from all LLM operations."""
    total_cost = 0.0
    total_ideas = 0

    for result in results:
        if hasattr(result, "generated_ideas"):
            for idea in result.generated_ideas:
                if hasattr(idea, "metadata") and idea.metadata:
                    cost = idea.metadata.get("llm_cost", 0)
                    total_cost += cost
                    total_ideas += 1

    if total_cost > 0:
        print_section("Cost Summary")
        print(f"💰 Total LLM Cost: ${total_cost:.4f}")
        print(f"📊 Total Ideas Generated: {total_ideas}")
        print(f"📈 Average Cost per Idea: ${total_cost/total_ideas:.4f}")
    else:
        print_section("Cost Summary")
        print("💰 No LLM costs incurred (likely using fallback methods)")


async def main():
    """Run the complete LLM Inductive Agent demonstration."""
    print("🧠 LLM Inductive Agent Demonstration")
    print("    Showcasing AI-powered pattern synthesis and insight generation")
    print("    This demo compares template-based vs LLM-powered inductive reasoning")

    # Collect results for cost summary
    results = []

    try:
        # Run all demonstrations
        await demo_basic_pattern_synthesis()
        await demo_scientific_discovery()
        await demo_business_intelligence()
        await demo_method_comparison()
        await demo_complex_synthesis()
        await demo_error_handling()

        print_section("Demonstration Complete")
        print("✅ All inductive reasoning demonstrations completed successfully!")
        print("\n🔍 Key Observations:")
        print("   • LLM agents provide context-aware, sophisticated pattern synthesis")
        print("   • Multiple inductive methods offer different cognitive approaches")
        print(
            "   • Error handling ensures graceful degradation when services unavailable"
        )
        print("   • Cost tracking enables budget-conscious AI utilization")
        print("   • Template agents provide reliable fallback functionality")

    except KeyboardInterrupt:
        print("\n\n⏹️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during demonstration: {e}")
        logger.error(f"Demo error: {e}", exc_info=True)

    # Show cost summary regardless of success/failure
    print_cost_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
