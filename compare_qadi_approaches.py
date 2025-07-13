#!/usr/bin/env python3
"""
QADI Comparison - Compare simple prompt vs full multi-agent system
Usage: uv run python compare_qadi_approaches.py "Your question here"

This script runs both approaches side-by-side to demonstrate the differences.
"""
import asyncio
import sys
import os
from pathlib import Path
import time

# Load .env
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value.strip('"').strip("'")

sys.path.insert(0, str(Path(__file__).parent / 'src'))

async def run_simple_prompt_approach(prompt: str):
    """Run the simple single-prompt approach."""
    from mad_spark_alt.core.llm_provider import setup_llm_providers, llm_manager, LLMRequest
    
    google_key = os.getenv('GOOGLE_API_KEY')
    if not google_key:
        return None, "No Google API key"
    
    await setup_llm_providers(google_api_key=google_key)
    
    qadi_prompt = f"""Analyze this question using the QADI methodology:
"{prompt}"

Provide exactly 3 practical answers based on:
1. One key question to explore
2. One creative hypothesis  
3. One logical deduction

Format:
QUESTION: [Your question]
HYPOTHESIS: [Your hypothesis]
DEDUCTION: [Your logical deduction]
ANSWER1: [First practical answer based on the question]
ANSWER2: [Second practical answer based on the hypothesis]
ANSWER3: [Third practical answer based on the deduction]"""

    start_time = time.time()
    
    request = LLMRequest(
        user_prompt=qadi_prompt,
        max_tokens=1000,
        temperature=0.7
    )
    
    try:
        response = await asyncio.wait_for(llm_manager.generate(request), timeout=30)
        execution_time = time.time() - start_time
        
        # Parse response
        content = response.content
        lines = content.split('\n')
        
        results = {
            'execution_time': execution_time,
            'cost': response.cost,
            'content': content,
            'approach': 'Single Prompt',
            'agent_count': 1,
            'thinking_phases': 1,
            'total_ideas': 3  # Fixed output
        }
        
        return results, None
        
    except Exception as e:
        return None, str(e)

async def run_multi_agent_approach(prompt: str):
    """Run the full multi-agent approach."""
    from mad_spark_alt.core.smart_orchestrator import SmartQADIOrchestrator
    from mad_spark_alt.core.llm_provider import setup_llm_providers
    
    google_key = os.getenv('GOOGLE_API_KEY')
    if not google_key:
        return None, "No Google API key"
    
    await setup_llm_providers(google_api_key=google_key)
    
    orchestrator = SmartQADIOrchestrator(auto_setup=True)
    
    # Optimized config for fair comparison
    cycle_config = {
        "max_ideas_per_method": 2,
        "require_reasoning": True,
    }
    
    start_time = time.time()
    
    try:
        result = await orchestrator.run_qadi_cycle(
            problem_statement=prompt,
            context="Generate practical, actionable insights",
            cycle_config=cycle_config
        )
        
        execution_time = time.time() - start_time
        
        # Count actual ideas generated
        total_ideas = len(result.synthesized_ideas)
        
        # Count unique agent types used
        unique_agents = len(set(result.agent_types.values()))
        
        # Extract key insights
        insights = []
        for phase_name, phase_result in result.phases.items():
            if phase_result and phase_result.generated_ideas:
                for idea in phase_result.generated_ideas:
                    insights.append(f"{phase_name}: {idea.content}")
        
        results = {
            'execution_time': execution_time,
            'cost': result.llm_cost,
            'content': '\n'.join(insights),
            'approach': 'Multi-Agent QADI',
            'agent_count': unique_agents,
            'thinking_phases': len(result.phases),
            'total_ideas': total_ideas,
            'agent_types': result.agent_types,
            'conclusion': result.conclusion.summary if result.conclusion else None
        }
        
        return results, None
        
    except Exception as e:
        return None, str(e)

async def compare_approaches(prompt: str):
    """Compare both approaches side by side."""
    print(f"📝 Comparing QADI approaches for: {prompt}")
    print("=" * 80)
    
    # Run both approaches
    print("\n🚀 Running Simple Prompt Approach...")
    simple_result, simple_error = await run_simple_prompt_approach(prompt)
    
    print("\n🧠 Running Multi-Agent Approach...")
    multi_result, multi_error = await run_multi_agent_approach(prompt)
    
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    if simple_error:
        print(f"❌ Simple approach failed: {simple_error}")
    else:
        print(f"\n📝 SIMPLE PROMPT APPROACH:")
        print(f"   ⏱️  Time: {simple_result['execution_time']:.1f}s")
        print(f"   💰 Cost: ${simple_result['cost']:.4f}")
        print(f"   🤖 Agents: {simple_result['agent_count']}")
        print(f"   🔄 Phases: {simple_result['thinking_phases']}")
        print(f"   💡 Ideas: {simple_result['total_ideas']}")
        print(f"   📊 Approach: Single LLM call with structured prompt")
    
    if multi_error:
        print(f"❌ Multi-agent approach failed: {multi_error}")
    else:
        print(f"\n🧠 MULTI-AGENT APPROACH:")
        print(f"   ⏱️  Time: {multi_result['execution_time']:.1f}s")
        print(f"   💰 Cost: ${multi_result['cost']:.4f}")
        print(f"   🤖 Agents: {multi_result['agent_count']}")
        print(f"   🔄 Phases: {multi_result['thinking_phases']}")
        print(f"   💡 Ideas: {multi_result['total_ideas']}")
        print(f"   📊 Approach: Specialized agents with sequential reasoning")
    
    # Quality comparison
    if simple_result and multi_result:
        print(f"\n📈 PERFORMANCE COMPARISON:")
        time_ratio = multi_result['execution_time'] / simple_result['execution_time']
        cost_ratio = multi_result['cost'] / simple_result['cost'] if simple_result['cost'] > 0 else float('inf')
        idea_ratio = multi_result['total_ideas'] / simple_result['total_ideas']
        
        print(f"   ⏱️  Time ratio: {time_ratio:.1f}x (multi-agent vs simple)")
        print(f"   💰 Cost ratio: {cost_ratio:.1f}x (multi-agent vs simple)")
        print(f"   💡 Idea ratio: {idea_ratio:.1f}x (multi-agent vs simple)")
        
        print(f"\n🎯 KEY DIFFERENCES:")
        print(f"   Simple Approach:")
        print(f"   • Single LLM call, faster execution")
        print(f"   • Fixed 3-answer format, limited flexibility")
        print(f"   • No specialized reasoning per phase")
        print(f"   • Essentially a formatted prompt to Gemini")
        
        print(f"\n   Multi-Agent Approach:")
        print(f"   • Multiple specialized agents with unique perspectives")
        print(f"   • Sequential reasoning building on previous phases")
        print(f"   • Adaptive agent selection (LLM vs template fallback)")
        print(f"   • Rich metadata, reasoning trails, and extensibility")
        print(f"   • Can integrate genetic evolution, custom evaluators")
        print(f"   • Structured conclusion synthesis")
        
        if multi_result.get('agent_types'):
            print(f"\n   Agent Types Used:")
            for method, agent_type in multi_result['agent_types'].items():
                print(f"   • {method}: {agent_type}")
    
    # Display actual outputs for comparison
    if simple_result:
        print(f"\n" + "="*50)
        print("SIMPLE APPROACH OUTPUT:")
        print("="*50)
        print(simple_result['content'][:500] + "..." if len(simple_result['content']) > 500 else simple_result['content'])
    
    if multi_result:
        print(f"\n" + "="*50)
        print("MULTI-AGENT APPROACH OUTPUT:")
        print("="*50)
        if multi_result.get('conclusion'):
            print(f"Conclusion: {multi_result['conclusion']}")
        print(multi_result['content'][:500] + "..." if len(multi_result['content']) > 500 else multi_result['content'])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: uv run python compare_qadi_approaches.py "Your question"')
        print('\nExamples:')
        print('  uv run python compare_qadi_approaches.py "how to improve team productivity"')
        print('  uv run python compare_qadi_approaches.py "sustainable energy solutions"')
    else:
        prompt = " ".join(sys.argv[1:])
        asyncio.run(compare_approaches(prompt))