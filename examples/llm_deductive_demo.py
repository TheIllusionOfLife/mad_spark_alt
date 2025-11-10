#!/usr/bin/env python3
"""
Demonstration of LLM-Powered Deductive Agent.

This script shows how the LLM Deductive Agent generates sophisticated logical analyses
using AI reasoning across multiple deductive frameworks.

Usage:
    python examples/llm_deductive_demo.py
"""

import asyncio
import os
from typing import Optional

from mad_spark_alt.agents.deduction.llm_agent import LLMDeductiveAgent
from mad_spark_alt.core.interfaces import IdeaGenerationRequest
from mad_spark_alt.core.llm_provider import LLMProvider, setup_llm_providers


async def demo_basic_logical_analysis():
    """Demonstrate basic logical analysis with the LLM Deductive Agent."""
    print("🔍 LLM Deductive Agent - Basic Logical Analysis Demo")
    print("=" * 60)

    # Check if we have API key
    google_key = os.getenv("GOOGLE_API_KEY")

    if not google_key:
        print("⚠️  No LLM API key found in environment variables.")
        print("   Please set GOOGLE_API_KEY to see AI-powered analysis.")
        print("   Skipping LLM demonstration...")
        return

    # Setup LLM provider
    await setup_llm_providers(google_api_key=google_key)

    # Create the agent
    agent = LLMDeductiveAgent(preferred_provider=LLMProvider.GOOGLE)

    # Test problem for logical analysis
    problem_statement = """
    A software development team claims their new authentication system is secure 
    because it uses 256-bit encryption and multi-factor authentication. However, 
    security audits have revealed several successful breach attempts through 
    social engineering and privilege escalation attacks.
    """

    context = """
    The system handles financial transactions and personal data for millions of users. 
    The team followed industry best practices for cryptographic implementation, 
    but human factors and system architecture may present vulnerabilities that 
    technical measures alone cannot address.
    """

    print(f"Problem: {problem_statement.strip()}")
    print(f"Context: {context.strip()}")
    print("\n🧠 Generating logical analyses using AI deductive reasoning...")

    # Create generation request
    request = IdeaGenerationRequest(
        problem_statement=problem_statement,
        context=context,
        max_ideas_per_method=6,
        generation_config={
            "logical_depth": "deep",
            "validation_rigor": "high",
            "include_counterarguments": True,
            "max_frameworks": 4,
        },
    )

    try:
        # Generate logical analyses
        result = await agent.generate_ideas(request)

        print(
            f"\n✅ Generated {len(result.generated_ideas)} logical analyses in {result.execution_time:.2f}s"
        )
        print(
            f"📊 Frameworks used: {', '.join(result.generation_metadata.get('frameworks_used', []))}"
        )

        # Display each analysis
        for i, analysis in enumerate(result.generated_ideas, 1):
            print(f"\n{'='*50}")
            print(f"📋 Logical Analysis #{i}")
            print(f"Framework: {analysis.metadata.get('framework', 'unknown')}")
            print(f"Confidence: {analysis.confidence_score:.2f}")
            print(f"\n💡 {analysis.content}")
            print(f"\n🔗 Reasoning Chain: {analysis.reasoning}")

            if analysis.metadata.get("premises"):
                print(f"\n📝 Premises: {analysis.metadata['premises']}")

            if analysis.metadata.get("implications"):
                print(f"\n➡️  Implications: {analysis.metadata['implications']}")

            if analysis.metadata.get("validation_criteria"):
                print(
                    f"\n✓ Validation Criteria: {analysis.metadata['validation_criteria']}"
                )

        # Show cost information
        total_cost = sum(a.metadata.get("llm_cost", 0) for a in result.generated_ideas)
        print(f"\n💰 Total LLM Cost: ${total_cost:.4f}")

    except Exception as e:
        print(f"❌ Error during logical analysis generation: {e}")


async def demo_framework_comparison():
    """Demonstrate different deductive frameworks in action."""
    print("\n\n🎯 Framework Comparison Demo")
    print("=" * 60)

    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not (openai_key or anthropic_key):
        print("⚠️  Skipping framework comparison demo - no API keys available.")
        return

    # Setup LLM providers
    await setup_llm_providers(
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
    )

    preferred_provider = LLMProvider.OPENAI if openai_key else LLMProvider.ANTHROPIC
    agent = LLMDeductiveAgent(preferred_provider=preferred_provider)

    # Different problem contexts to test framework selection
    test_cases = [
        {
            "name": "Formal Logic Problem",
            "problem": "Prove that if all premises in a logical argument are true and the reasoning is valid, then the conclusion must be true.",
            "config": {"max_frameworks": 3, "formal_logic": True},
            "expected_frameworks": ["logical_validation", "proof_construction"],
        },
        {
            "name": "Requirement Analysis Problem",
            "problem": "A mobile app must work offline, sync when online, handle conflicts, and maintain data integrity across devices.",
            "config": {"max_frameworks": 3, "systematic_analysis": True},
            "expected_frameworks": [
                "requirement_validation",
                "systematic_decomposition",
            ],
        },
        {
            "name": "Constraint Validation Problem",
            "problem": "Design a recommendation system that must process 1M+ users, respond in <100ms, use <2GB RAM, and achieve >95% accuracy.",
            "config": {"max_frameworks": 2, "constraint_validation": True},
            "expected_frameworks": ["constraint_analysis"],
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
            frameworks_used = result.generation_metadata.get("frameworks_used", [])

            print(f"✅ Frameworks selected: {', '.join(frameworks_used)}")
            print(f"📝 Generated {len(result.generated_ideas)} logical analyses")

            # Show one example analysis
            if result.generated_ideas:
                example = result.generated_ideas[0]
                print(f"💡 Example: {example.content[:100]}...")

        except Exception as e:
            print(f"❌ Error: {e}")


async def demo_template_vs_llm_comparison():
    """Compare template-based vs LLM-powered logical analysis."""
    print("\n\n⚡ Template vs LLM Comparison")
    print("=" * 60)

    from mad_spark_alt.agents.deduction.agent import DeductionAgent

    # Test with same problem
    problem = "A database system must guarantee ACID properties while scaling to handle millions of concurrent transactions."

    # Template-based agent
    template_agent = DeductionAgent()
    print("📝 Template-based Deductive Agent:")

    request = IdeaGenerationRequest(
        problem_statement=problem,
        max_ideas_per_method=3,
    )

    template_result = await template_agent.generate_ideas(request)

    for i, idea in enumerate(template_result.generated_ideas, 1):
        print(f"  {i}. {idea.content}")

    # LLM-powered agent (if available)
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if openai_key or anthropic_key:
        await setup_llm_providers(
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
        )

        preferred_provider = LLMProvider.OPENAI if openai_key else LLMProvider.ANTHROPIC
        llm_agent = LLMDeductiveAgent(preferred_provider=preferred_provider)

        print(f"\n🧠 LLM-powered Deductive Agent:")

        llm_result = await llm_agent.generate_ideas(request)

        for i, idea in enumerate(llm_result.generated_ideas, 1):
            framework = idea.metadata.get("framework", "unknown")
            confidence = idea.metadata.get("confidence_level", "unknown")
            print(f"  {i}. [{framework}:{confidence}] {idea.content}")

        print(f"\n📊 Comparison:")
        print(
            f"  Template Agent: {len(template_result.generated_ideas)} analyses in {template_result.execution_time:.3f}s"
        )
        print(
            f"  LLM Agent: {len(llm_result.generated_ideas)} analyses in {llm_result.execution_time:.3f}s"
        )

        if llm_result.generated_ideas:
            total_cost = sum(
                a.metadata.get("llm_cost", 0) for a in llm_result.generated_ideas
            )
            print(f"  LLM Cost: ${total_cost:.4f}")
    else:
        print("\n🧠 LLM-powered agent skipped (no API keys)")


async def demo_logical_rigor_showcase():
    """Showcase the logical rigor and systematic analysis capabilities."""
    print("\n\n🎓 Logical Rigor Showcase")
    print("=" * 60)

    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not (openai_key or anthropic_key):
        print("⚠️  Skipping logical rigor showcase - no API keys available.")
        return

    # Setup LLM providers
    await setup_llm_providers(
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
    )

    preferred_provider = LLMProvider.OPENAI if openai_key else LLMProvider.ANTHROPIC
    agent = LLMDeductiveAgent(preferred_provider=preferred_provider)

    # Complex logical problem requiring rigorous analysis
    problem = """
    A distributed consensus algorithm claims to achieve Byzantine fault tolerance 
    with optimal performance. The algorithm must handle up to f malicious nodes 
    out of 3f+1 total nodes, guarantee safety and liveness properties, and 
    maintain consistent state across all honest nodes even under network partitions.
    """

    context = """
    The algorithm operates in an asynchronous network with possible message delays, 
    duplications, and losses. Malicious nodes can exhibit arbitrary behavior including 
    sending conflicting messages, remaining silent, or coordinating attacks. 
    The system must distinguish between network failures and malicious behavior.
    """

    print("🧮 Complex Problem Requiring Rigorous Logical Analysis:")
    print(f"Problem: {problem.strip()}")
    print(f"Context: {context.strip()}")

    request = IdeaGenerationRequest(
        problem_statement=problem,
        context=context,
        max_ideas_per_method=4,
        generation_config={
            "logical_depth": "very_deep",
            "validation_rigor": "highest",
            "formal_logic": True,
            "include_proof_structure": True,
            "max_frameworks": 3,
        },
    )

    try:
        result = await agent.generate_ideas(request)

        print(f"\n🔬 Rigorous Analysis Results:")
        print(f"Generated {len(result.generated_ideas)} systematic analyses")

        for i, analysis in enumerate(result.generated_ideas, 1):
            framework = analysis.metadata.get("framework", "unknown")
            confidence = analysis.metadata.get("confidence_level", "unknown")

            print(
                f"\n📐 Analysis #{i} - {framework.replace('_', ' ').title()} (Confidence: {confidence})"
            )
            print(f"Analysis: {analysis.content}")

            if analysis.metadata.get("validation_criteria"):
                print(f"Validation: {analysis.metadata['validation_criteria']}")

    except Exception as e:
        print(f"❌ Error during rigorous analysis: {e}")


async def main():
    """Run all demonstration scenarios."""
    print("🚀 LLM Deductive Agent Demonstration")
    print("=" * 60)
    print("This demo shows how the LLM Deductive Agent generates sophisticated")
    print("logical analyses using AI reasoning across multiple deductive frameworks.")
    print()

    try:
        await demo_basic_logical_analysis()
        await demo_framework_comparison()
        await demo_template_vs_llm_comparison()
        await demo_logical_rigor_showcase()

        print("\n🎉 Demo completed successfully!")
        print("\nThe LLM Deductive Agent demonstrates:")
        print("• Systematic logical validation and analysis")
        print("• Multiple deductive reasoning frameworks")
        print("• Intelligent framework selection based on problem type")
        print("• Rigorous logical reasoning with structured proofs")
        print("• Cost-effective AI integration with precise reasoning")

    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
