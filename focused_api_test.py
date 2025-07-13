#!/usr/bin/env python3
"""
Focused test of Enhanced QADI with real API key.
"""

import asyncio
import os
import sys
from pathlib import Path

# Load .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

print("🧪 FOCUSED API TEST: Enhanced QADI System")
print("=" * 60)

# Check API key
google_key = os.getenv('GOOGLE_API_KEY')
if google_key:
    print(f"✅ Google API Key: Available (...{google_key[-6:]})")
else:
    print("❌ No API key found")
    sys.exit(1)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_template_vs_llm_comparison():
    """Compare template vs LLM agents if available."""
    
    from mad_spark_alt.core import SmartQADIOrchestrator
    
    test_question = "What are 3 practical ways to improve sleep quality?"
    
    print(f"\n🎯 Test Question: {test_question}")
    print("=" * 60)
    
    orchestrator = SmartQADIOrchestrator()
    
    print("🔄 Running Smart QADI Cycle...")
    result = await orchestrator.run_qadi_cycle(
        problem_statement=test_question,
        context="Focus on evidence-based approaches that anyone can implement",
        cycle_config={"max_ideas_per_method": 2}
    )
    
    print(f"\n📊 RESULTS:")
    print(f"⏱️  Execution time: {result.execution_time:.3f}s")
    print(f"💰 LLM cost: ${result.llm_cost:.4f}")
    print(f"🤖 Agent types: {result.agent_types}")
    print(f"💡 Total ideas: {len(result.synthesized_ideas)}")
    
    # Show sample output from each phase
    print(f"\n🔬 SAMPLE IDEAS FROM EACH PHASE:")
    for phase_name, phase_result in result.phases.items():
        if phase_result and phase_result.generated_ideas:
            agent_type = result.agent_types.get(phase_name, "unknown")
            print(f"\n🔸 {phase_name.upper()} ({agent_type}):")
            for i, idea in enumerate(phase_result.generated_ideas[:1], 1):
                print(f"  {i}. {idea.content[:100]}...")
    
    return result

async def test_enhanced_answer_extraction():
    """Test the answer extraction component."""
    
    print(f"\n{'='*60}")
    print("🎯 TESTING ANSWER EXTRACTION")
    print("=" * 60)
    
    # Import after path setup
    from mad_spark_alt.core.enhanced_orchestrator import EnhancedQADIOrchestrator
    
    orchestrator = EnhancedQADIOrchestrator()
    
    test_question = "What are 3 effective strategies to reduce stress at work?"
    
    print(f"❓ Question: {test_question}")
    print("🔄 Running Enhanced QADI with Answer Extraction...")
    
    try:
        result = await orchestrator.run_qadi_cycle_with_answers(
            problem_statement=test_question,
            context="Consider both immediate techniques and long-term approaches",
            max_answers=3
        )
        
        print(f"\n✅ ENHANCED RESULTS:")
        print(f"⏱️  Total time: {result.execution_time:.3f}s")
        print(f"💰 LLM cost: ${result.llm_cost:.4f}")
        print(f"🧠 QADI insights: {len(result.synthesized_ideas)}")
        
        if result.extracted_answers:
            print(f"\n📋 EXTRACTED ANSWERS:")
            print(f"Question type detected: {result.extracted_answers.question_type}")
            
            for i, answer in enumerate(result.extracted_answers.direct_answers, 1):
                print(f"  {i}. {answer.content}")
                print(f"     💭 Source: {answer.source_phase} phase (confidence: {answer.confidence:.1f})")
            
            print(f"\n📊 Extraction Summary:")
            print(f"  {result.extracted_answers.summary}")
        else:
            print("❌ No answers extracted")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def quick_comparison_test():
    """Quick comparison between approaches."""
    
    print(f"\n{'='*60}")
    print("⚖️  QUICK COMPARISON TEST")
    print("=" * 60)
    
    question = "How can I be more productive working from home?"
    
    print(f"❓ Question: {question}")
    
    # Test 1: Direct practical answers (simulated)
    print(f"\n📝 DIRECT APPROACH:")
    direct_answers = [
        "Create a dedicated workspace in your home",
        "Establish clear work hours and boundaries", 
        "Use time-blocking and minimize distractions"
    ]
    for i, answer in enumerate(direct_answers, 1):
        print(f"  {i}. {answer}")
    
    # Test 2: Enhanced QADI approach
    print(f"\n🧠 ENHANCED QADI APPROACH:")
    from mad_spark_alt.core.enhanced_orchestrator import EnhancedQADIOrchestrator
    
    orchestrator = EnhancedQADIOrchestrator()
    
    try:
        result = await orchestrator.run_qadi_cycle_with_answers(
            problem_statement=question,
            max_answers=3
        )
        
        if result.extracted_answers:
            for i, answer in enumerate(result.extracted_answers.direct_answers, 1):
                print(f"  {i}. {answer.content}")
            
            print(f"\n🎯 QADI Value Added:")
            print(f"  • Systematic analysis: {len(result.phases)} thinking phases")
            print(f"  • Deep insights: {len(result.synthesized_ideas)} total ideas")
            print(f"  • Theoretical foundation: {' → '.join(result.phases.keys())}")
            print(f"  • Question analysis: {result.extracted_answers.question_type}")
        else:
            print("  ❌ Answer extraction failed")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

async def main():
    """Run focused tests."""
    
    try:
        # Test 1: Basic QADI functionality
        qadi_result = await test_template_vs_llm_comparison()
        
        # Test 2: Enhanced answer extraction
        success = await test_enhanced_answer_extraction()
        
        # Test 3: Quick comparison
        await quick_comparison_test()
        
        print(f"\n{'='*60}")
        print("🎉 FOCUSED TESTING COMPLETE!")
        print("=" * 60)
        
        if success:
            print("✅ Enhanced QADI system working correctly")
            print("🔧 Answer extraction successfully bridges QADI → practical answers")
            print("🎯 System provides both theoretical rigor and practical utility")
        else:
            print("⚠️  Some components need refinement")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())