#!/usr/bin/env python3
"""
Real API test for enhanced semantic operators.
Tests the implementation with actual Google API calls.
"""

import asyncio
import os
import sys
from typing import Dict

import pytest

# Add src to path
sys.path.insert(0, 'src')

from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
from mad_spark_alt.core.llm_provider import GoogleProvider
from mad_spark_alt.evolution.interfaces import EvaluationContext
from mad_spark_alt.evolution.semantic_operators import BatchSemanticMutationOperator


@pytest.mark.integration
async def test_enhanced_semantic_operators():
    """Test enhanced semantic operators with real API calls."""
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment")
        return False
        
    print("🚀 Testing Enhanced Semantic Operators with Real Google API")
    print("=" * 60)
    
    # Initialize LLM provider
    provider = GoogleProvider(api_key=api_key)
    operator = BatchSemanticMutationOperator(provider)
    
    # Create test evaluation context
    evaluation_context = EvaluationContext(
        original_question="How can we reduce plastic waste in our community?",
        current_best_scores={
            "impact": 0.7,
            "feasibility": 0.4,  # Weak score - should be targeted
            "accessibility": 0.3,  # Very weak score - should be targeted
            "sustainability": 0.8,
            "scalability": 0.6
        },
        target_improvements=["feasibility", "accessibility"]
    )
    
    # Test 1: Regular idea with semantic mutation
    print("\n🧪 Test 1: Regular Idea - Standard Semantic Mutation")
    print("-" * 50)
    
    regular_idea = GeneratedIdea(
        content="Create a community recycling program with collection points",
        thinking_method=ThinkingMethod.ABDUCTION,
        agent_name="TestAgent",
        generation_prompt="How to reduce waste",
        confidence_score=0.7,
        reasoning="Community engagement approach",
        metadata={"generation": 0}
    )
    
    try:
        result1 = await operator.mutate_single(regular_idea, evaluation_context)
        
        print(f"✅ Original: {regular_idea.content[:80]}...")
        print(f"✅ Mutated:  {result1.content[:80]}...")
        print(f"✅ Operator: {result1.metadata.get('operator')}")
        print(f"✅ Mutation Type: {result1.metadata.get('mutation_type')}")
        print(f"✅ Is Breakthrough: {result1.metadata.get('is_breakthrough')}")
        print(f"✅ Cost: ${result1.metadata.get('llm_cost', 0):.4f}")
        
        # Verify it's not a breakthrough mutation
        assert result1.metadata.get('is_breakthrough') == False
        print("✅ Correctly identified as regular mutation")
        
    except Exception as e:
        print(f"❌ Regular mutation test failed: {e}")
        return False
    
    # Test 2: High-scoring idea with breakthrough mutation
    print("\n🧪 Test 2: High-Scoring Idea - Breakthrough Mutation")
    print("-" * 50)
    
    high_scoring_idea = GeneratedIdea(
        content="AI-powered smart waste sorting system with IoT sensors",
        thinking_method=ThinkingMethod.ABDUCTION,
        agent_name="TestAgent",
        generation_prompt="Advanced waste management",
        confidence_score=0.9,
        reasoning="High-tech solution with proven effectiveness",
        metadata={
            "generation": 2,
            "overall_fitness": 0.88,  # High fitness - should trigger breakthrough
            "overall_fitness": 0.85
        }
    )
    
    try:
        result2 = await operator.mutate_single(high_scoring_idea, evaluation_context)
        
        print(f"✅ Original: {high_scoring_idea.content[:80]}...")
        print(f"✅ Mutated:  {result2.content[:80]}...")
        print(f"✅ Operator: {result2.metadata.get('operator')}")
        print(f"✅ Mutation Type: {result2.metadata.get('mutation_type')}")
        print(f"✅ Is Breakthrough: {result2.metadata.get('is_breakthrough')}")
        print(f"✅ Cost: ${result2.metadata.get('llm_cost', 0):.4f}")
        
        # Verify it's a breakthrough mutation
        assert result2.metadata.get('is_breakthrough') == True
        assert "breakthrough" in result2.metadata.get('operator', '').lower()
        print("✅ Correctly identified as breakthrough mutation")
        
        # Check for revolutionary content indicators
        content_lower = result2.content.lower()
        revolutionary_indicators = [
            'advanced', 'revolutionary', 'cutting-edge', 'innovative',
            'breakthrough', 'quantum', 'ai', 'smart', 'predictive',
            'blockchain', 'machine learning', 'automated'
        ]
        
        found_indicators = [word for word in revolutionary_indicators if word in content_lower]
        print(f"✅ Revolutionary indicators found: {found_indicators}")
        
        if len(found_indicators) >= 2:
            print("✅ Content shows revolutionary characteristics")
        else:
            print("⚠️  Content may not be sufficiently revolutionary")
            
    except Exception as e:
        print(f"❌ Breakthrough mutation test failed: {e}")
        return False
    
    # Test 3: Batch mutation with mixed scoring
    print("\n🧪 Test 3: Batch Mutation with Mixed Scoring")
    print("-" * 50)
    
    mixed_ideas = [
        GeneratedIdea(
            content="Community composting program",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="TestAgent",
            generation_prompt="Organic waste reduction",
            metadata={"generation": 0}
        ),
        GeneratedIdea(
            content="Advanced plastic-to-fuel conversion facility",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="TestAgent", 
            generation_prompt="Advanced recycling",
            metadata={"generation": 1, "overall_fitness": 0.82}  # High scoring
        )
    ]
    
    try:
        batch_results = await operator.mutate_batch(mixed_ideas, evaluation_context)
        
        print(f"✅ Processed {len(batch_results)} ideas in batch")
        
        for i, result in enumerate(batch_results):
            print(f"✅ Idea {i+1}:")
            print(f"   Original: {mixed_ideas[i].content[:60]}...")
            print(f"   Mutated:  {result.content[:60]}...")
            print(f"   Breakthrough: {result.metadata.get('is_breakthrough')}")
            print(f"   Cost: ${result.metadata.get('llm_cost', 0):.4f}")
            
    except Exception as e:
        print(f"❌ Batch mutation test failed: {e}")
        return False
    
    # Test 4: Context targeting verification
    print("\n🧪 Test 4: Context Targeting Verification")
    print("-" * 50)
    
    # Create context focused on sustainability
    sustainability_context = EvaluationContext(
        original_question="How can we create environmentally sustainable solutions?",
        current_best_scores={
            "sustainability": 0.2,  # Very weak
            "impact": 0.7,
            "feasibility": 0.6,
            "accessibility": 0.5,
            "scalability": 0.4
        },
        target_improvements=["sustainability"]
    )
    
    sustainability_idea = GeneratedIdea(
        content="Solar-powered recycling centers in neighborhoods",
        thinking_method=ThinkingMethod.ABDUCTION,
        agent_name="TestAgent",
        generation_prompt="Renewable energy integration",
        metadata={"generation": 0}
    )
    
    try:
        result4 = await operator.mutate_single(sustainability_idea, sustainability_context)
        
        print(f"✅ Original: {sustainability_idea.content}")
        print(f"✅ Mutated:  {result4.content}")
        
        # Check if the mutation addresses sustainability
        content_lower = result4.content.lower()
        sustainability_keywords = [
            'sustainable', 'environment', 'renewable', 'green', 'eco',
            'carbon', 'emission', 'biodegradable', 'clean energy',
            'solar', 'wind', 'recycl', 'compost', 'organic'
        ]
        
        found_keywords = [word for word in sustainability_keywords if word in content_lower]
        print(f"✅ Sustainability keywords found: {found_keywords}")
        
        if len(found_keywords) >= 2:
            print("✅ Mutation successfully targets sustainability")
        else:
            print("⚠️  Mutation may not be sufficiently focused on sustainability")
            
    except Exception as e:
        print(f"❌ Context targeting test failed: {e}")
        return False
    
    print("\n🎉 All Enhanced Semantic Operator Tests Completed Successfully!")
    print("=" * 60)
    
    # Calculate total costs
    total_cost = sum([
        result1.metadata.get('llm_cost', 0),
        result2.metadata.get('llm_cost', 0),
        result4.metadata.get('llm_cost', 0)
    ]) + sum(result.metadata.get('llm_cost', 0) for result in batch_results)
    
    print(f"💰 Total API cost for tests: ${total_cost:.4f}")
    print("✅ Enhanced semantic operators working correctly with real API!")
    
    return True


async def main():
    """Main test function."""
    try:
        success = await test_enhanced_semantic_operators()
        if success:
            print("\n🎯 RESULT: All tests passed - Enhanced semantic operators ready for production!")
            sys.exit(0)
        else:
            print("\n❌ RESULT: Some tests failed - Check implementation")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    asyncio.run(main())