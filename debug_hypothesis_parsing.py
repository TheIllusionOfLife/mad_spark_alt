#!/usr/bin/env python3
"""Debug hypothesis parsing with real LLM responses."""

import asyncio
import os
import logging
from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator
from mad_spark_alt.core.llm_provider import setup_llm_providers

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

async def test_hypothesis_parsing():
    """Test hypothesis parsing with real LLM calls."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("GOOGLE_API_KEY not found")
        return
    
    # Setup LLM providers
    await setup_llm_providers(google_api_key)
    
    # Test with the problematic query
    orchestrator = SimpleQADIOrchestrator(num_hypotheses=3)
    
    try:
        result = await orchestrator.run_qadi_cycle(
            "How can we reduce plastic waste?", 
            "Focus on practical solutions"
        )
        
        print(f"\n✅ SUCCESS: Got {len(result.hypotheses)} hypotheses")
        for i, h in enumerate(result.hypotheses):
            print(f"H{i+1}: {h[:150]}...")
            
    except Exception as e:
        print(f"\n❌ FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(test_hypothesis_parsing())