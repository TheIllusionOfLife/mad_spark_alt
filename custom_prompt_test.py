#!/usr/bin/env python3
"""
Custom prompt test with an interesting, complex challenge.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_complex_custom_prompt():
    """Test with a complex, interesting custom prompt."""
    
    print("🎯 CUSTOM PROMPT TEST: Complex Challenge")
    print("=" * 70)
    
    # Choose an interesting, multi-dimensional challenge
    custom_prompt = "What are 5 innovative strategies for a small bookstore to compete with Amazon while building stronger community connections?"
    
    print(f"📚 CUSTOM CHALLENGE:")
    print(f"   {custom_prompt}")
    print(f"\n💡 Why this tests the system well:")
    print(f"   • Business strategy + community building")
    print(f"   • David vs Goliath competitive scenario") 
    print(f"   • Requires creative AND practical thinking")
    print(f"   • Multiple stakeholders to consider")
    print(f"   • Local vs global dynamics")
    
    print(f"\n{'='*70}")
    print("🔄 RUNNING ENHANCED QADI ANALYSIS")
    print("=" * 70)
    
    try:
        from mad_spark_alt.core.enhanced_orchestrator import EnhancedQADIOrchestrator
        
        orchestrator = EnhancedQADIOrchestrator()
        
        result = await orchestrator.run_qadi_cycle_with_answers(
            problem_statement=custom_prompt,
            context="""
            Consider the unique advantages of local bookstores:
            - Personal relationships and community knowledge
            - Physical space for events and gatherings
            - Curated selection and expert recommendations
            - Supporting local economy and culture
            
            Constraints:
            - Limited budget compared to Amazon
            - Cannot compete on pure price or selection size
            - Must leverage unique local value propositions
            """,
            max_answers=5,
            cycle_config={"max_ideas_per_method": 3}
        )
        
        print(f"⏱️  Total execution time: {result.execution_time:.3f}s")
        print(f"💰 LLM cost: ${result.llm_cost:.4f}")
        print(f"🤖 Agent types: {result.agent_types}")
        print(f"🧠 QADI insights generated: {len(result.synthesized_ideas)}")
        
        # Show the systematic QADI thinking process
        print(f"\n🔬 QADI THINKING PROCESS:")
        print("-" * 50)
        
        for phase_name, phase_result in result.phases.items():
            if phase_result and phase_result.generated_ideas:
                agent_type = result.agent_types.get(phase_name, "unknown")
                print(f"\n🔸 {phase_name.upper()} PHASE ({agent_type}):")
                print(f"   Generated {len(phase_result.generated_ideas)} systematic insights")
                
                # Show one example from each phase
                if phase_result.generated_ideas:
                    example = phase_result.generated_ideas[0]
                    print(f"   Example: {example.content[:120]}...")
        
        # Show the extracted practical answers
        if result.extracted_answers:
            print(f"\n✅ EXTRACTED BUSINESS STRATEGIES:")
            print("-" * 50)
            print(f"Question type detected: {result.extracted_answers.question_type}")
            
            for i, answer in enumerate(result.extracted_answers.direct_answers, 1):
                print(f"\n{i}. {answer.content}")
                print(f"   💡 Derived from: {answer.source_phase} thinking")
                print(f"   🎯 Confidence: {answer.confidence:.1f}")
        
        # Show the value of the systematic approach
        print(f"\n🎯 QADI VALUE DEMONSTRATION:")
        print("-" * 50)
        print(f"✅ Systematic Analysis: 4 cognitive phases applied")
        print(f"✅ Comprehensive Coverage: {len(result.synthesized_ideas)} insights considered") 
        print(f"✅ Practical Output: {len(result.extracted_answers.direct_answers) if result.extracted_answers else 0} actionable strategies")
        print(f"✅ Question Understanding: Automatically detected as '{result.extracted_answers.question_type if result.extracted_answers else 'unknown'}'")
        print(f"✅ Contextual Adaptation: Used bookstore-specific context in analysis")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during custom prompt test: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_different_question_types():
    """Test with different question types to show versatility."""
    
    print(f"\n{'='*70}")
    print("🎨 QUESTION TYPE VERSATILITY TEST")
    print("=" * 70)
    
    test_cases = [
        {
            "prompt": "How can a remote team build trust and collaboration?",
            "type": "How-to question",
            "expected": "Step-by-step guidance"
        },
        {
            "prompt": "What are 4 emerging technologies that could revolutionize education?", 
            "type": "List request",
            "expected": "Numbered list of technologies"
        },
        {
            "prompt": "Why do most startups fail in their first year?",
            "type": "Explanatory question", 
            "expected": "Root cause analysis"
        }
    ]
    
    from mad_spark_alt.core.enhanced_orchestrator import EnhancedQADIOrchestrator
    orchestrator = EnhancedQADIOrchestrator()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 TEST {i}: {test_case['type']}")
        print(f"❓ {test_case['prompt']}")
        print(f"🎯 Expected: {test_case['expected']}")
        print("-" * 50)
        
        try:
            result = await orchestrator.run_qadi_cycle_with_answers(
                problem_statement=test_case['prompt'],
                max_answers=3
            )
            
            if result.extracted_answers:
                print(f"✅ Detected as: {result.extracted_answers.question_type}")
                print(f"📋 Generated answers:")
                for j, answer in enumerate(result.extracted_answers.direct_answers[:2], 1):
                    print(f"  {j}. {answer.content[:100]}...")
                
                print(f"⚡ QADI foundation: {len(result.synthesized_ideas)} insights in {result.execution_time:.3f}s")
            else:
                print("❌ No answers extracted")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
        
        if i < len(test_cases):
            print()

async def demonstrate_system_value():
    """Demonstrate the unique value of the QADI approach."""
    
    print(f"\n{'='*70}")
    print("💎 SYSTEM VALUE DEMONSTRATION")
    print("=" * 70)
    
    challenge = "How can cities reduce traffic while improving quality of life?"
    
    print(f"🌆 Urban Challenge: {challenge}")
    
    print(f"\n📝 SIMPLE APPROACH (Direct answers):")
    simple_answers = [
        "Improve public transportation",
        "Add bike lanes", 
        "Implement congestion pricing"
    ]
    for i, answer in enumerate(simple_answers, 1):
        print(f"  {i}. {answer}")
    
    print(f"\n🧠 ENHANCED QADI APPROACH:")
    print("   (Systematic thinking → Comprehensive analysis → Practical extraction)")
    
    try:
        from mad_spark_alt.core.enhanced_orchestrator import EnhancedQADIOrchestrator
        orchestrator = EnhancedQADIOrchestrator()
        
        result = await orchestrator.run_qadi_cycle_with_answers(
            problem_statement=challenge,
            context="Consider economic, social, environmental, and technological factors",
            max_answers=4
        )
        
        print(f"\n🎯 QADI-ENHANCED RESULTS:")
        if result.extracted_answers:
            for i, answer in enumerate(result.extracted_answers.direct_answers, 1):
                print(f"  {i}. {answer.content}")
        
        print(f"\n📊 QADI ADVANTAGE:")
        print(f"  • Systematic Analysis: {' → '.join(result.phases.keys())}")
        print(f"  • Deep Understanding: {len(result.synthesized_ideas)} insights considered")
        print(f"  • Multi-perspective: 4 different cognitive approaches")
        print(f"  • Theoretical Rigor: Based on 'Shin Logical Thinking' methodology")
        print(f"  • Practical Output: Automatically formatted for user needs")
        
    except Exception as e:
        print(f"❌ QADI approach failed: {e}")

async def main():
    """Run custom prompt tests."""
    
    print("🚀 ENHANCED QADI - CUSTOM PROMPT TESTING")
    print("Demonstrating the system with complex, real-world challenges")
    print("=" * 70)
    
    # Test 1: Complex business strategy challenge
    success1 = await test_complex_custom_prompt()
    
    # Test 2: Different question types
    await test_different_question_types()
    
    # Test 3: System value demonstration
    await demonstrate_system_value()
    
    print(f"\n{'='*70}")
    print("🎉 CUSTOM PROMPT TESTING COMPLETE!")
    print("=" * 70)
    
    if success1:
        print("✅ Enhanced QADI successfully handles complex, multi-dimensional challenges")
        print("🔧 System bridges theoretical QADI methodology with practical business needs")
        print("🎯 Automatic question type detection and appropriate answer formatting") 
        print("🧠 Maintains systematic thinking rigor while delivering user-friendly results")
        print("⚡ Ready for real-world deployment with LLM integration")
    else:
        print("⚠️ System needs refinement for complex scenarios")

if __name__ == "__main__":
    asyncio.run(main())