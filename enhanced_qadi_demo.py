#!/usr/bin/env python3
"""
Demonstration of the Enhanced QADI system with Answer Extraction.

This shows how the system now bridges the gap between abstract QADI thinking
and direct, actionable user answers.
"""

import asyncio
import sys
import os

# Add src to path so we can import our new modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mad_spark_alt.core.enhanced_orchestrator import EnhancedQADIOrchestrator

async def demo_enhanced_qadi():
    """Demonstrate the enhanced QADI system with answer extraction."""
    
    print("🚀 Enhanced QADI System Demo")
    print("Bridging the gap: QADI Insights → Direct Answers")
    print("=" * 70)
    
    orchestrator = EnhancedQADIOrchestrator()
    
    # Test different types of user questions
    test_questions = [
        {
            "question": "What are 3 practical ways to reduce food waste at home?",
            "expected": "List of 3 specific, actionable methods",
            "context": "Focus on everyday solutions for individuals"
        },
        {
            "question": "How can I improve my productivity while working from home?",
            "expected": "Step-by-step guidance and strategies",
            "context": "Consider environment, habits, and tools"
        },
        {
            "question": "What are 5 ways to make cities more sustainable?",
            "expected": "List of 5 urban sustainability approaches",
            "context": "Consider technology, policy, and infrastructure"
        }
    ]
    
    for i, test_case in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"🧪 TEST {i}: {test_case['question']}")
        print(f"🎯 Expected: {test_case['expected']}")
        print(f"📝 Context: {test_case['context']}")
        print("=" * 70)
        
        # Run enhanced QADI cycle
        result = await orchestrator.run_qadi_cycle_with_answers(
            problem_statement=test_case["question"],
            context=test_case["context"],
            max_answers=5  # Extract up to 5 answers
        )
        
        # Display results using the enhanced display method
        orchestrator.display_enhanced_results(result)
        
        # Show the improvement
        if result.extracted_answers:
            print(f"\n✅ SUCCESS METRICS:")
            print(f"   • Question Type: {result.extracted_answers.question_type}")
            print(f"   • Answers Extracted: {len(result.extracted_answers.direct_answers)}")
            print(f"   • Total QADI Ideas: {result.extracted_answers.total_qadi_ideas}")
            print(f"   • Extraction Time: {result.answer_extraction_time:.3f}s")
            print(f"   • Extraction Method: {result.extracted_answers.extraction_method}")
        
        print(f"\n🎉 User gets direct answers instead of abstract QADI patterns!")
        
        if i < len(test_questions):
            print(f"\n⏳ Next test in 2 seconds...")
            await asyncio.sleep(2)

async def demo_quick_answers():
    """Demonstrate the convenience method for quick answers."""
    
    print(f"\n{'='*70}")
    print("🏃‍♂️ QUICK ANSWERS DEMO")
    print("Using convenience method for rapid results")
    print("=" * 70)
    
    orchestrator = EnhancedQADIOrchestrator()
    
    question = "How to reduce plastic waste in oceans?"
    
    print(f"❓ Question: {question}")
    print("⚡ Getting direct answers...")
    
    answers = await orchestrator.get_direct_answers(
        question=question,
        context="Focus on both prevention and cleanup approaches",
        max_answers=4
    )
    
    print(f"\n✅ DIRECT ANSWERS:")
    for i, answer in enumerate(answers, 1):
        print(f"{i}. {answer}")
    
    print(f"\n🎯 Perfect for quick consultations and rapid ideation!")

async def compare_old_vs_new():
    """Compare old QADI vs new Enhanced QADI side by side."""
    
    print(f"\n{'='*70}")
    print("⚖️  OLD vs NEW COMPARISON")
    print("=" * 70)
    
    from mad_spark_alt.core import SmartQADIOrchestrator
    
    question = "What are 3 ways to improve online learning?"
    
    print(f"❓ Question: {question}")
    
    # OLD APPROACH
    print(f"\n🔸 OLD QADI APPROACH:")
    print("-" * 35)
    
    old_orchestrator = SmartQADIOrchestrator()
    old_result = await old_orchestrator.run_qadi_cycle(
        problem_statement=question,
        cycle_config={"max_ideas_per_method": 2}
    )
    
    print("Raw QADI Output:")
    for phase, result in old_result.phases.items():
        if result and result.generated_ideas:
            print(f"  {phase}: {len(result.generated_ideas)} abstract ideas")
            for idea in result.generated_ideas[:1]:
                print(f"    - {idea.content[:80]}...")
    
    # NEW APPROACH
    print(f"\n🔸 NEW ENHANCED APPROACH:")
    print("-" * 35)
    
    new_orchestrator = EnhancedQADIOrchestrator()
    new_result = await new_orchestrator.run_qadi_cycle_with_answers(
        problem_statement=question,
        max_answers=3
    )
    
    if new_result.extracted_answers:
        print("Direct Answers:")
        for i, answer in enumerate(new_result.extracted_answers.direct_answers, 1):
            print(f"  {i}. {answer.content}")
    
    print(f"\n🎯 IMPROVEMENT:")
    print("   ❌ Old: Abstract meta-thinking about solutions")
    print("   ✅ New: Concrete, actionable answers")
    print("   🚀 Same QADI rigor + User-friendly output!")

async def main():
    """Run all demonstrations."""
    try:
        await demo_enhanced_qadi()
        await demo_quick_answers()
        await compare_old_vs_new()
        
        print(f"\n{'='*70}")
        print("🎉 ENHANCED QADI SYSTEM DEMONSTRATION COMPLETE!")
        print("=" * 70)
        print("✅ Problem Solved: QADI now produces direct answers")
        print("🔧 Solution: Answer extraction bridges abstract → concrete")
        print("🎯 Benefit: Users get actionable responses to their questions")
        print("🧠 Maintained: Full QADI thinking rigor and methodology")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())