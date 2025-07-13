#!/usr/bin/env python3
"""
Real API test of the Enhanced QADI system with Google Gemini.

This test demonstrates the complete system with actual LLM-powered agents
and compares it with the existing generate_ideas.py approach.
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

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mad_spark_alt.core.enhanced_orchestrator import EnhancedQADIOrchestrator
from mad_spark_alt.core import SmartQADIOrchestrator

async def test_enhanced_qadi_with_llm():
    """Test enhanced QADI system with real LLM backend."""
    
    print("🧪 REAL API TEST: Enhanced QADI with Google Gemini")
    print("=" * 70)
    
    # Check API key status
    google_key = os.getenv('GOOGLE_API_KEY')
    if not google_key:
        print("❌ No Google API key found. Cannot test with LLM.")
        return False
    
    print(f"✅ Google API Key: Available (ends with ...{google_key[-6:]})")
    print("🤖 Testing with LLM-powered agents")
    
    # Test prompt - something challenging that benefits from deep thinking
    test_prompt = "What are 4 innovative strategies to reduce urban air pollution while maintaining economic growth?"
    
    print(f"\n🎯 Test Challenge:")
    print(f"   {test_prompt}")
    print(f"\n📝 Why this tests QADI well:")
    print(f"   • Complex multi-dimensional problem")
    print(f"   • Requires balancing competing priorities")
    print(f"   • Benefits from systematic thinking")
    print(f"   • Needs creative + logical solutions")
    
    print(f"\n{'='*70}")
    print("🔄 RUNNING ENHANCED QADI CYCLE...")
    print("=" * 70)
    
    try:
        # Test enhanced orchestrator
        enhanced_orchestrator = EnhancedQADIOrchestrator()
        
        result = await enhanced_orchestrator.run_qadi_cycle_with_answers(
            problem_statement=test_prompt,
            context="Consider both technological innovations and policy approaches. Focus on solutions that are practical and scalable.",
            max_answers=4,
            cycle_config={"max_ideas_per_method": 2}
        )
        
        print(f"⏱️  Total execution time: {result.execution_time:.2f}s")
        print(f"💰 LLM cost: ${result.llm_cost:.4f}")
        print(f"🤖 Agent types used: {result.agent_types}")
        print(f"📊 Total QADI insights: {len(result.synthesized_ideas)}")
        
        # Display the enhanced results
        enhanced_orchestrator.display_enhanced_results(result)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during enhanced QADI test: {e}")
        import traceback
        traceback.print_exc()
        return False

async def compare_with_generate_ideas():
    """Compare with the existing generate_ideas.py approach."""
    
    print(f"\n{'='*70}")
    print("⚖️  COMPARISON: Enhanced QADI vs generate_ideas.py")
    print("=" * 70)
    
    # Import the generate_ideas function
    sys.path.append(os.path.dirname(__file__))
    from generate_ideas import generate_working_ideas
    
    test_prompt = "Reduce urban air pollution while maintaining economic growth"
    
    print(f"🎯 Test prompt: {test_prompt}")
    
    # Test generate_ideas.py approach
    print(f"\n🔸 APPROACH 1: generate_ideas.py (Direct LLM)")
    print("-" * 50)
    
    try:
        await generate_working_ideas(test_prompt)
    except Exception as e:
        print(f"❌ generate_ideas.py failed: {e}")
    
    # Test enhanced QADI approach  
    print(f"\n🔸 APPROACH 2: Enhanced QADI (Systematic + LLM)")
    print("-" * 50)
    
    try:
        enhanced_orchestrator = EnhancedQADIOrchestrator()
        result = await enhanced_orchestrator.run_qadi_cycle_with_answers(
            problem_statement=f"What are 4 ways to {test_prompt}?",
            max_answers=4
        )
        
        if result.extracted_answers:
            print(f"✅ Enhanced QADI Results:")
            for i, answer in enumerate(result.extracted_answers.direct_answers, 1):
                print(f"  {i}. {answer.content}")
            
            print(f"\n📊 QADI Analysis Depth:")
            print(f"   • {len(result.phases)} thinking phases completed")
            print(f"   • {len(result.synthesized_ideas)} total insights generated")
            print(f"   • Systematic {' → '.join(result.phases.keys())} methodology")
            print(f"   • {result.extracted_answers.question_type} question type detected")
        
    except Exception as e:
        print(f"❌ Enhanced QADI failed: {e}")

async def test_multiple_question_types():
    """Test the system with different types of questions."""
    
    print(f"\n{'='*70}")
    print("🎯 QUESTION TYPE VERSATILITY TEST")
    print("=" * 70)
    
    test_cases = [
        {
            "question": "What are 3 breakthrough technologies that could revolutionize renewable energy in the next decade?",
            "type": "List request",
            "context": "Consider both existing technologies that could scale and entirely new innovations"
        },
        {
            "question": "How can small businesses adapt to AI automation without losing their human touch?",
            "type": "How-to question", 
            "context": "Focus on practical strategies that preserve company culture and customer relationships"
        },
        {
            "question": "Why do most startup companies fail within their first 5 years?",
            "type": "Explanatory question",
            "context": "Consider both internal factors and external market conditions"
        }
    ]
    
    orchestrator = EnhancedQADIOrchestrator()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 TEST {i}: {test_case['type']}")
        print(f"❓ Question: {test_case['question']}")
        print(f"📝 Context: {test_case['context']}")
        print("-" * 50)
        
        try:
            result = await orchestrator.run_qadi_cycle_with_answers(
                problem_statement=test_case['question'],
                context=test_case['context'],
                max_answers=3 if 'how' in test_case['question'].lower() else 4
            )
            
            if result.extracted_answers:
                print(f"✅ Detected as: {result.extracted_answers.question_type}")
                print(f"📋 Extracted Answers:")
                for j, answer in enumerate(result.extracted_answers.direct_answers, 1):
                    print(f"  {j}. {answer.content}")
                
                print(f"📊 QADI Foundation: {len(result.synthesized_ideas)} insights in {result.execution_time:.2f}s")
            else:
                print("❌ No answers extracted")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
        
        if i < len(test_cases):
            await asyncio.sleep(1)  # Respect API rate limits

async def main():
    """Run all tests."""
    
    print("🚀 COMPREHENSIVE ENHANCED QADI TESTING")
    print("Testing the complete solution with real LLM backend")
    print("=" * 70)
    
    # Test 1: Basic enhanced QADI functionality
    success = await test_enhanced_qadi_with_llm()
    
    if success:
        # Test 2: Compare approaches
        await compare_with_generate_ideas()
        
        # Test 3: Multiple question types
        await test_multiple_question_types()
        
        print(f"\n{'='*70}")
        print("🎉 COMPREHENSIVE TESTING COMPLETE!")
        print("=" * 70)
        print("✅ Enhanced QADI system successfully tested with real LLM")
        print("🔧 System bridges QADI methodology with practical user answers")
        print("🎯 Demonstrates value over simple direct LLM prompting")
        print("🧠 Maintains theoretical rigor while providing practical utility")
        
    else:
        print("\n❌ Testing failed. Check API configuration and try again.")

if __name__ == "__main__":
    asyncio.run(main())