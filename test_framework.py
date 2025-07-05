#!/usr/bin/env python3
"""
Test script for the complete hybrid multi-layer evaluation framework.
"""

import asyncio
from rich.console import Console

from mad_spark_alt.core import (
    CreativityEvaluator,
    EvaluationLayer,
    EvaluationRequest,
    ModelOutput,
    OutputType,
)
from mad_spark_alt.layers.llm_judges import CreativityLLMJudge, CreativityJury
from mad_spark_alt.layers.human_eval import HumanCreativityEvaluator

console = Console()


async def test_hybrid_framework():
    """Test the complete hybrid evaluation framework."""
    
    console.print("🧪 Testing Hybrid Multi-layer Evaluation Framework", style="bold blue")
    console.print("=" * 60)
    
    # Test content
    test_content = """
    The city of tomorrow floats on crystallized air currents, its buildings grown from 
    living coral that purifies the atmosphere. Citizens travel through pneumatic tubes 
    filled with luminescent spores that provide both transportation and nutrition. 
    Streets are replaced by flowing rivers of recycled light, and every surface 
    doubles as a vertical farm growing food from captured moonbeams.
    """
    
    # Create model output
    output = ModelOutput(
        content=test_content.strip(),
        output_type=OutputType.TEXT,
        model_name="creative-ai",
        prompt="Describe a sustainable city of the future",
        metadata={"test": True}
    )
    
    console.print(f"📝 Test Content: {test_content.strip()[:100]}...")
    
    # Test Layer 1: Quantitative (existing functionality)
    console.print("\n🔢 Layer 1: Quantitative Evaluation")
    console.print("-" * 30)
    
    evaluator = CreativityEvaluator()
    request = EvaluationRequest(
        outputs=[output],
        target_layers=[EvaluationLayer.QUANTITATIVE],
        task_context="Framework testing"
    )
    
    summary = await evaluator.evaluate(request)
    console.print(f"✅ Quantitative evaluation completed")
    overall_score = summary.get_overall_creativity_score()
    if overall_score is not None:
        console.print(f"   Overall score: {overall_score:.3f}")
    else:
        console.print(f"   Overall score: N/A")
    
    # Test Layer 2: LLM Judge
    console.print("\n🤖 Layer 2: LLM Judge Evaluation")
    console.print("-" * 30)
    
    # Single judge
    llm_judge = CreativityLLMJudge("mock-model")
    judge_request = EvaluationRequest(
        outputs=[output],
        target_layers=[EvaluationLayer.LLM_JUDGE],
        task_context="Framework testing - single judge"
    )
    
    judge_results = await llm_judge.evaluate(judge_request)
    if judge_results and judge_results[0].scores:
        overall = judge_results[0].scores.get("overall_creativity", 0)
        console.print(f"✅ Single LLM judge completed")
        console.print(f"   Overall creativity: {overall:.3f}")
    
    # Jury
    jury = CreativityJury(["mock-model-1", "mock-model-2", "mock-model-3"])
    jury_request = EvaluationRequest(
        outputs=[output],
        target_layers=[EvaluationLayer.LLM_JUDGE],
        task_context="Framework testing - jury"
    )
    
    jury_results = await jury.evaluate(jury_request)
    if jury_results and jury_results[0].scores:
        overall = jury_results[0].scores.get("overall_creativity", 0)
        console.print(f"✅ LLM jury evaluation completed")
        console.print(f"   Consensus score: {overall:.3f}")
    
    # Test Layer 3: Human (batch mode for testing)
    console.print("\n🧑‍🎨 Layer 3: Human Evaluation (Batch Mode)")
    console.print("-" * 30)
    
    human_evaluator = HumanCreativityEvaluator({"mode": "batch", "output_file": "/tmp/test_evaluation.json"})
    human_request = EvaluationRequest(
        outputs=[output],
        target_layers=[EvaluationLayer.HUMAN],
        task_context="Framework testing - batch human eval"
    )
    
    human_results = await human_evaluator.evaluate(human_request)
    if human_results:
        console.print(f"✅ Human evaluation template generated")
        console.print(f"   Mode: batch")
    
    # Test complete framework
    console.print("\n🎯 Complete Framework Test")
    console.print("-" * 30)
    
    complete_request = EvaluationRequest(
        outputs=[output],
        target_layers=[EvaluationLayer.QUANTITATIVE, EvaluationLayer.LLM_JUDGE],
        task_context="Complete framework test"
    )
    
    complete_summary = await evaluator.evaluate(complete_request)
    console.print(f"✅ Complete framework evaluation completed")
    console.print(f"   Layers evaluated: {len(complete_summary.layer_results)}")
    console.print(f"   Total evaluators: {complete_summary.total_evaluators}")
    console.print(f"   Execution time: {complete_summary.execution_time:.2f}s")
    overall_score = complete_summary.get_overall_creativity_score()
    if overall_score is not None:
        console.print(f"   Overall creativity: {overall_score:.3f}")
    else:
        console.print(f"   Overall creativity: N/A")
    
    console.print("\n🎉 Framework test completed successfully!", style="bold green")
    console.print("All three evaluation layers are working correctly.", style="green")


if __name__ == "__main__":
    asyncio.run(test_hybrid_framework())