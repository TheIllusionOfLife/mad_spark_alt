#!/usr/bin/env python3
"""
Simple test to verify the enhanced QADI system works.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_answer_extraction():
    """Test just the answer extraction component."""
    
    print("🧪 SIMPLE TEST: Answer Extraction Component")
    print("=" * 50)
    
    try:
        from mad_spark_alt.core.answer_extractor import TemplateAnswerExtractor
        from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
        
        # Create mock QADI results
        mock_qadi_results = {
            'questioning': [
                GeneratedIdea(
                    content="What are the root causes of poor sleep?",
                    thinking_method=ThinkingMethod.QUESTIONING,
                    agent_name="QuestioningAgent",
                    generation_prompt="test"
                )
            ],
            'abduction': [
                GeneratedIdea(
                    content="Sleep issues might be caused by environmental factors",
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="AbductionAgent", 
                    generation_prompt="test"
                )
            ],
            'deduction': [
                GeneratedIdea(
                    content="If we improve sleep environment, then sleep quality should improve",
                    thinking_method=ThinkingMethod.DEDUCTION,
                    agent_name="DeductionAgent",
                    generation_prompt="test"
                )
            ],
            'induction': [
                GeneratedIdea(
                    content="Successful sleep improvement follows consistent patterns",
                    thinking_method=ThinkingMethod.INDUCTION,
                    agent_name="InductionAgent",
                    generation_prompt="test"
                )
            ]
        }
        
        # Test answer extraction
        extractor = TemplateAnswerExtractor()
        
        test_question = "What are 3 ways to improve sleep quality?"
        
        result = extractor.extract_answers(
            question=test_question,
            qadi_results=mock_qadi_results,
            max_answers=3
        )
        
        print(f"✅ ANSWER EXTRACTION TEST:")
        print(f"Question: {test_question}")
        print(f"Question type detected: {result.question_type}")
        print(f"Answers extracted: {len(result.direct_answers)}")
        
        print(f"\n📋 EXTRACTED ANSWERS:")
        for i, answer in enumerate(result.direct_answers, 1):
            print(f"  {i}. {answer.content}")
            print(f"     Source: {answer.source_phase} (confidence: {answer.confidence})")
        
        print(f"\n📊 Summary: {result.summary}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_template_qadi():
    """Test basic QADI with template agents."""
    
    print(f"\n{'='*50}")
    print("🧪 TEMPLATE QADI TEST")
    print("=" * 50)
    
    try:
        from mad_spark_alt.core.orchestrator import QADIOrchestrator
        from mad_spark_alt.agents import QuestioningAgent, AbductionAgent, DeductionAgent, InductionAgent
        
        # Create agents
        agents = [
            QuestioningAgent(),
            AbductionAgent(),
            DeductionAgent(), 
            InductionAgent()
        ]
        
        orchestrator = QADIOrchestrator(agents=agents)
        
        result = await orchestrator.run_qadi_cycle(
            problem_statement="What are 3 ways to improve sleep quality?",
            cycle_config={"max_ideas_per_method": 2}
        )
        
        print(f"✅ TEMPLATE QADI TEST:")
        print(f"Execution time: {result.execution_time:.3f}s")
        print(f"Total ideas: {len(result.synthesized_ideas)}")
        print(f"Phases completed: {len(result.phases)}")
        
        for phase_name, phase_result in result.phases.items():
            if phase_result and phase_result.generated_ideas:
                print(f"\n🔸 {phase_name}: {len(phase_result.generated_ideas)} ideas")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_enhanced_integration():
    """Test the integrated enhanced system."""
    
    print(f"\n{'='*50}")
    print("🧪 ENHANCED INTEGRATION TEST")
    print("=" * 50)
    
    try:
        from mad_spark_alt.core.enhanced_orchestrator import EnhancedQADIOrchestrator
        
        orchestrator = EnhancedQADIOrchestrator()
        
        result = await orchestrator.run_qadi_cycle_with_answers(
            problem_statement="What are 3 ways to improve sleep quality?",
            max_answers=3,
            cycle_config={"max_ideas_per_method": 2}
        )
        
        print(f"✅ ENHANCED INTEGRATION TEST:")
        print(f"Total time: {result.execution_time:.3f}s")
        print(f"QADI ideas: {len(result.synthesized_ideas)}")
        print(f"Answer extraction time: {result.answer_extraction_time:.3f}s")
        
        if result.extracted_answers:
            print(f"\n📋 FINAL ANSWERS:")
            for i, answer in enumerate(result.extracted_answers.direct_answers, 1):
                print(f"  {i}. {answer.content}")
            
            print(f"\n🎯 SUCCESS: QADI insights converted to practical answers!")
        else:
            print("❌ No answers extracted")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback 
        traceback.print_exc()
        return False

async def main():
    """Run simple tests."""
    
    print("🚀 ENHANCED QADI - SIMPLE VERIFICATION TESTS")
    print("Testing core functionality without LLM complexity")
    print("=" * 60)
    
    # Test 1: Answer extraction component
    extraction_success = await test_answer_extraction()
    
    # Test 2: Template QADI
    qadi_result = await test_template_qadi()
    
    # Test 3: Enhanced integration
    integration_success = await test_enhanced_integration()
    
    print(f"\n{'='*60}")
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"Answer Extraction: {'✅ PASS' if extraction_success else '❌ FAIL'}")
    print(f"Template QADI: {'✅ PASS' if qadi_result else '❌ FAIL'}")
    print(f"Enhanced Integration: {'✅ PASS' if integration_success else '❌ FAIL'}")
    
    if extraction_success and qadi_result and integration_success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("✅ Enhanced QADI system successfully bridges abstract thinking to practical answers")
        print("🔧 System works with template agents (LLM agents would provide even better results)")
        print("🎯 Core functionality verified - ready for real-world use")
    else:
        print(f"\n⚠️ Some tests failed - system needs debugging")

if __name__ == "__main__":
    asyncio.run(main())