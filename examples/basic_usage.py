"""
Basic usage examples for Mad Spark Alt.
"""

import asyncio
from src.mad_spark_alt import CreativityEvaluator, EvaluationRequest, ModelOutput, OutputType


async def basic_evaluation_example():
    """Example of evaluating a single AI output."""
    
    # Create a model output to evaluate
    output = ModelOutput(
        content="The quantum cat simultaneously existed in a state of both curiosity and indifference, pondering the philosophical implications of Schrödinger's litter box while composing haikus about the uncertainty principle.",
        output_type=OutputType.TEXT,
        model_name="example-model",
        prompt="Write a creative story about a cat and quantum physics"
    )
    
    # Create evaluation request
    request = EvaluationRequest(
        outputs=[output],
        task_context="Creative writing task"
    )
    
    # Create evaluator and run evaluation
    evaluator = CreativityEvaluator()
    summary = await evaluator.evaluate(request)
    
    # Print results
    print(f"Overall creativity score: {summary.get_overall_creativity_score():.3f}")
    print(f"Execution time: {summary.execution_time:.2f}s")
    
    for layer, results in summary.layer_results.items():
        print(f"\n{layer.value.title()} Layer Results:")
        for result in results:
            print(f"  {result.evaluator_name}:")
            for metric, score in result.scores.items():
                print(f"    {metric}: {score:.3f}")


async def diversity_comparison_example():
    """Example of comparing multiple outputs for diversity."""
    
    # Create multiple outputs to compare
    outputs = [
        ModelOutput(
            content="AI will revolutionize healthcare by enabling personalized treatments.",
            output_type=OutputType.TEXT,
            model_name="model-a"
        ),
        ModelOutput(
            content="Artificial intelligence transforms medical care through predictive analytics.",
            output_type=OutputType.TEXT,
            model_name="model-b"
        ),
        ModelOutput(
            content="The purple elephant danced magnificently under the moonlit sky.",
            output_type=OutputType.TEXT,
            model_name="model-c"
        ),
    ]
    
    # Create evaluation request
    request = EvaluationRequest(
        outputs=outputs,
        task_context="Comparing AI responses about healthcare"
    )
    
    # Evaluate
    evaluator = CreativityEvaluator()
    summary = await evaluator.evaluate(request)
    
    # Show diversity metrics
    print("Diversity Comparison Results:")
    for i, result in enumerate(summary.layer_results.get(OutputType.TEXT, [])):
        if result.evaluator_name == "diversity_evaluator":
            print(f"\nOutput {i+1}:")
            print(f"  Novelty Score: {result.scores.get('novelty_score', 0):.3f}")
            print(f"  Semantic Uniqueness: {result.scores.get('semantic_uniqueness', 0):.3f}")
            print(f"  Lexical Diversity: {result.scores.get('lexical_diversity', 0):.3f}")


async def code_evaluation_example():
    """Example of evaluating code creativity."""
    
    code_output = ModelOutput(
        content="""
def fibonacci_poetry(n):
    '''Generate Fibonacci numbers with poetic variable names.'''
    harmony, melody = 0, 1
    for verse in range(n):
        print(f"Verse {verse}: {harmony}")
        harmony, melody = melody, harmony + melody
    return "The mathematical symphony concludes."

# A creative approach to the classic algorithm
fibonacci_poetry(10)
        """,
        output_type=OutputType.CODE,
        model_name="code-model",
        prompt="Write a creative implementation of the Fibonacci sequence"
    )
    
    request = EvaluationRequest(
        outputs=[code_output],
        task_context="Creative coding task"
    )
    
    evaluator = CreativityEvaluator()
    summary = await evaluator.evaluate(request)
    
    print("Code Creativity Evaluation:")
    print(f"Overall score: {summary.get_overall_creativity_score():.3f}")
    
    for layer, results in summary.layer_results.items():
        for result in results:
            if "code" in result.scores:
                print(f"\nCode Quality Metrics:")
                for metric, score in result.scores.items():
                    if "code" in metric:
                        print(f"  {metric}: {score:.3f}")


if __name__ == "__main__":
    print("Running basic evaluation example...")
    asyncio.run(basic_evaluation_example())
    
    print("\n" + "="*50)
    print("Running diversity comparison example...")
    asyncio.run(diversity_comparison_example())
    
    print("\n" + "="*50)
    print("Running code evaluation example...")
    asyncio.run(code_evaluation_example())