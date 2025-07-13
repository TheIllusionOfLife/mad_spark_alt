#!/usr/bin/env python3
"""
Simple script to test Enhanced QADI with your custom prompt.

Usage: uv run python test_my_idea.py "Your custom prompt"
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_custom_prompt(prompt: str):
    """Test Enhanced QADI with a custom prompt."""
    
    print("🚀 Enhanced QADI - Custom Prompt Test")
    print("=" * 60)
    print(f"💡 Your Question: {prompt}")
    print("=" * 60)
    
    try:
        from mad_spark_alt.core.enhanced_orchestrator import EnhancedQADIOrchestrator
        
        orchestrator = EnhancedQADIOrchestrator()
        
        # Run enhanced QADI cycle
        print("\n🔄 Running QADI Analysis...")
        
        result = await orchestrator.run_qadi_cycle_with_answers(
            problem_statement=prompt,
            max_answers=5,  # Adjust based on your needs
            cycle_config={"max_ideas_per_method": 3}
        )
        
        # Display results
        print(f"\n⏱️  Processing time: {result.execution_time:.3f}s")
        print(f"🧠 QADI insights generated: {len(result.synthesized_ideas)}")
        print(f"🤖 Agent types used: {list(set(result.agent_types.values()))}")
        
        # Show extracted answers
        if result.extracted_answers:
            print(f"\n✅ DIRECT ANSWERS ({result.extracted_answers.question_type}):")
            print("-" * 50)
            
            for i, answer in enumerate(result.extracted_answers.direct_answers, 1):
                print(f"\n{i}. {answer.content}")
                print(f"   Source: {answer.source_phase} phase")
        
        # Show QADI thinking process summary
        print(f"\n🔬 QADI THINKING PROCESS:")
        print("-" * 50)
        for phase, phase_result in result.phases.items():
            if phase_result:
                print(f"• {phase.upper()}: {len(phase_result.generated_ideas)} insights")
        
        print(f"\n🎯 Analysis complete! The system used systematic thinking")
        print(f"   ({' → '.join(result.phases.keys())}) to generate answers.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point."""
    
    if len(sys.argv) < 2:
        print("🚀 Enhanced QADI - Test Your Ideas")
        print("\n📖 Usage:")
        print('  uv run python test_my_idea.py "your question or challenge"')
        print("\n🎯 Examples:")
        print('  • "What are 5 ways to improve customer service?"')
        print('  • "How can small businesses use AI effectively?"')
        print('  • "What strategies can help reduce workplace stress?"')
        sys.exit(1)
    
    # Get the prompt from command line
    custom_prompt = " ".join(sys.argv[1:])
    
    # Run the test
    asyncio.run(test_custom_prompt(custom_prompt))

if __name__ == "__main__":
    main()