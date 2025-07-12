#!/usr/bin/env python3
"""Test the robust QADI system with various challenging prompts."""

import asyncio
import os
from pathlib import Path

# Load .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

# Import directly to avoid circular import issues
from mad_spark_alt.core.robust_orchestrator import RobustQADIOrchestrator

# Test prompts that previously caused timeouts
TEST_PROMPTS = [
    "What is the best life?",
    "Suggest 5 ideas to spend 100 USD for the weekend",
    "How can we achieve world peace?",
    "What is the meaning of existence?",
    "Create a new philosophical framework",
]

async def test_robust_system():
    """Test the robust system with challenging prompts."""
    print("🧪 Testing Robust QADI System")
    print(f"✅ Google API Key: {'SET' if os.getenv('GOOGLE_API_KEY') else 'NOT SET'}")
    
    orchestrator = RobustQADIOrchestrator(
        default_timeout=120.0,  # 2 minutes total
        phase_timeout=25.0      # 25 seconds per phase
    )
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {prompt}")
        print('='*60)
        
        try:
            result = await orchestrator.run_qadi_cycle(
                problem_statement=prompt,
                context="Provide thoughtful and creative responses",
                cycle_config={
                    "max_ideas_per_method": 2,
                    "require_reasoning": False
                }
            )
            
            print(f"✅ Success! Completed in {result.execution_time:.2f}s")
            print(f"💰 Cost: ${result.llm_cost:.4f}")
            
            # Show phase summary
            for phase_name, phase_result in result.phases.items():
                ideas_count = len(phase_result.generated_ideas)
                status = "✅" if ideas_count > 0 else "⚠️"
                agent_type = result.agent_types.get(phase_name, "unknown")
                print(f"{status} {phase_name}: {ideas_count} ideas ({agent_type})")
                
                if phase_result.error_message:
                    print(f"   Error: {phase_result.error_message}")
            
            # Show timeout summary
            if result.metadata and "timeout_summary" in result.metadata:
                summary = result.metadata["timeout_summary"]
                print(f"\n⏱️  Timing: Total {summary['total_elapsed']:.1f}s of {summary['total_timeout']:.0f}s")
                
        except Exception as e:
            print(f"❌ Failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_robust_system())