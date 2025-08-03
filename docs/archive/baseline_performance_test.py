#!/usr/bin/env python3
"""Baseline performance test for evolution with 3 generations and 5 population."""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

from mad_spark_alt.cli import _run_evolution_pipeline
from mad_spark_alt.core.llm_provider import setup_llm_providers, llm_manager


async def run_baseline_test():
    """Run baseline performance test and collect metrics."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("GOOGLE_API_KEY not found")
        return
    
    # Setup LLM providers
    await setup_llm_providers(google_api_key)
    
    # Test configuration
    problem = "How can we reduce food waste in urban areas?"
    context = "Focus on practical, implementable solutions"
    generations = 3
    population = 5
    
    print(f"🧪 Running Baseline Performance Test")
    print(f"=" * 60)
    print(f"Problem: {problem}")
    print(f"Generations: {generations}")
    print(f"Population: {population}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"=" * 60)
    
    # Track metrics
    start_time = time.time()
    initial_llm_calls = llm_manager.get_total_calls() if hasattr(llm_manager, 'get_total_calls') else 0
    initial_cost = llm_manager.get_total_cost() if hasattr(llm_manager, 'get_total_cost') else 0.0
    
    # Run evolution
    output_file = f"baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        await _run_evolution_pipeline(
            problem=problem,
            context=context,
            quick=False,
            generations=generations,
            population=population,
            temperature=None,
            output_file=output_file
        )
        
        # Calculate metrics
        total_time = time.time() - start_time
        total_llm_calls = (llm_manager.get_total_calls() if hasattr(llm_manager, 'get_total_calls') else 0) - initial_llm_calls
        total_cost = (llm_manager.get_total_cost() if hasattr(llm_manager, 'get_total_cost') else 0.0) - initial_cost
        
        # Load results file to get additional metrics
        if Path(output_file).exists():
            with open(output_file, 'r') as f:
                results = json.load(f)
                evolution_metrics = results.get('metrics', {})
        else:
            evolution_metrics = {}
        
        # Create baseline metrics report
        baseline_metrics = {
            "test_config": {
                "generations": generations,
                "population": population,
                "problem": problem,
                "timestamp": datetime.now().isoformat()
            },
            "performance": {
                "total_time_seconds": round(total_time, 2),
                "total_time_minutes": round(total_time / 60, 2),
                "llm_calls": total_llm_calls,
                "total_cost": round(total_cost, 4),
                "cost_per_idea": round(total_cost / (generations * population) if generations * population > 0 else 0, 4)
            },
            "evolution_metrics": evolution_metrics,
            "breakdown": {
                "avg_time_per_generation": round(total_time / generations if generations > 0 else 0, 2),
                "avg_time_per_evaluation": round(total_time / (generations * population) if generations * population > 0 else 0, 2),
                "cache_hit_rate": evolution_metrics.get('cache_stats', {}).get('hit_rate', 0),
                "semantic_operators_used": evolution_metrics.get('semantic_operators_enabled', False)
            }
        }
        
        # Save baseline metrics
        baseline_file = "baseline_metrics.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline_metrics, f, indent=2)
        
        # Display results
        print(f"\n📊 Baseline Performance Metrics:")
        print(f"=" * 60)
        print(f"Total Time: {baseline_metrics['performance']['total_time_seconds']}s ({baseline_metrics['performance']['total_time_minutes']} minutes)")
        print(f"LLM Calls: {baseline_metrics['performance']['llm_calls']}")
        print(f"Total Cost: ${baseline_metrics['performance']['total_cost']}")
        print(f"Cost per Idea: ${baseline_metrics['performance']['cost_per_idea']}")
        print(f"Cache Hit Rate: {baseline_metrics['breakdown']['cache_hit_rate']:.1%}")
        print(f"Avg Time per Generation: {baseline_metrics['breakdown']['avg_time_per_generation']}s")
        print(f"Avg Time per Evaluation: {baseline_metrics['breakdown']['avg_time_per_evaluation']}s")
        print(f"\n✅ Baseline metrics saved to: {baseline_file}")
        print(f"✅ Full results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error during baseline test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_baseline_test())