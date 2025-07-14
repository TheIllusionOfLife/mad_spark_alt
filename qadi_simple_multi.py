#!/usr/bin/env python3
"""
QADI Simple Multi - Multiple simple LLM calls mimicking multi-agent approach
Usage: uv run python qadi_simple_multi.py "Your question here"

This version uses Google API with multiple simple calls to avoid timeouts
while providing multi-agent-like perspectives.
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

async def run_qadi_phase(phase_name: str, prompt: str, previous_insights: str = "", concrete_mode: bool = False, classification_result=None):
    """Run a single QADI phase using adaptive prompts based on question classification."""
    from mad_spark_alt.core.llm_provider import setup_llm_providers, llm_manager, LLMRequest
    from mad_spark_alt.core.adaptive_prompts import get_adaptive_prompt, get_complexity_adjusted_params
    
    # Get adaptive prompt if classification is available
    if classification_result:
        adaptive_prompt = get_adaptive_prompt(
            phase_name=phase_name,
            classification_result=classification_result,
            prompt=prompt,
            previous_insights=previous_insights,
            concrete_mode=concrete_mode
        )
        
        # Get complexity-adjusted parameters
        llm_params = get_complexity_adjusted_params(classification_result)
    else:
        # Fallback to static prompts if no classification
        regular_prompts = {
            "questioning": f"""As a questioning specialist, generate 2 insightful questions about: "{prompt}"
{previous_insights}
Format each question on a new line starting with "Q:".""",
            
            "abduction": f"""As a hypothesis specialist, generate 2 creative hypotheses about: "{prompt}"
{previous_insights}
Consider unexpected connections and possibilities.
Format each hypothesis on a new line starting with "H:".""",
            
            "deduction": f"""As a logical reasoning specialist, generate 2 logical deductions about: "{prompt}"
{previous_insights}
Apply systematic reasoning and derive conclusions.
Format each deduction on a new line starting with "D:".""",
            
            "induction": f"""As a pattern synthesis specialist, generate 2 pattern-based insights about: "{prompt}"
{previous_insights}
Identify recurring themes and general principles.
Format each insight on a new line starting with "I:"."""
        }
        
        concrete_prompts = {
            "questioning": f"""As an implementation specialist, generate 2 practical questions about: "{prompt}"
{previous_insights}
Focus on implementation challenges, resource requirements, and feasibility concerns.
Format each question on a new line starting with "Q:".""",
            
            "abduction": f"""As a solution architect, generate 2 specific, implementable solutions for: "{prompt}"
{previous_insights}
Provide concrete approaches with specific tools, methods, or technologies.
Include real-world examples where possible.
Format each solution on a new line starting with "H:".""",
            
            "deduction": f"""As a project planner, generate 2 logical implementation steps for: "{prompt}"
{previous_insights}
Focus on step-by-step approaches, prerequisites, and concrete actions.
Format each step on a new line starting with "D:".""",
            
            "induction": f"""As a best practices specialist, generate 2 concrete patterns or methodologies for: "{prompt}"
{previous_insights}
Identify proven approaches, specific frameworks, and actionable principles.
Format each pattern on a new line starting with "I:"."""
        }
        
        phase_prompts = concrete_prompts if concrete_mode else regular_prompts
        adaptive_prompt = phase_prompts[phase_name]
        llm_params = {"max_tokens": 300, "temperature": 0.7}
    
    request = LLMRequest(
        user_prompt=adaptive_prompt,
        max_tokens=llm_params["max_tokens"],
        temperature=llm_params["temperature"]
    )
    
    try:
        response = await asyncio.wait_for(llm_manager.generate(request), timeout=20)
        return response.content, response.cost, response.model
    except asyncio.TimeoutError:
        return f"[{phase_name} phase timed out]", 0.0, "unknown"
    except Exception as e:
        return f"[{phase_name} phase error: {e}]", 0.0, "unknown"

async def run_simple_multi_agent_qadi(prompt: str, concrete_mode: bool = False, question_type_override: str = None):
    """Run QADI using multiple simple LLM calls with adaptive prompts."""
    from mad_spark_alt.core.llm_provider import setup_llm_providers
    from mad_spark_alt.core.json_utils import format_llm_cost
    from mad_spark_alt.core.prompt_classifier import classify_question, QuestionType
    
    print(f"📝 {prompt}")
    print("🚀 QADI SIMPLE MULTI-AGENT (LLM Mode)")
    print("=" * 70)
    
    # Classify the question for adaptive prompts
    print("🧠 Analyzing question type...", end='', flush=True)
    classification_start = time.time()
    classification_result = classify_question(prompt)
    classification_time = time.time() - classification_start
    print(f" ✓ ({classification_time:.1f}s)")
    
    # Apply manual override if provided
    if question_type_override:
        try:
            override_type = QuestionType(question_type_override.lower())
            original_type = classification_result.question_type
            classification_result.question_type = override_type
            override_applied = True
        except ValueError:
            print(f"⚠️  Warning: Invalid question type '{question_type_override}', using auto-detected type")
            override_applied = False
    else:
        override_applied = False
    
    # Display classification results
    print(f"📊 Question Analysis:")
    if override_applied:
        print(f"  ├─ Type: {classification_result.question_type.value.title()} (manually set)")
        print(f"  ├─ Auto-detected: {original_type.value.title()}")
    else:
        print(f"  ├─ Type: {classification_result.question_type.value.title()}")
    print(f"  ├─ Complexity: {classification_result.complexity.value.title()}")
    print(f"  ├─ Confidence: {classification_result.confidence:.1%}")
    if classification_result.domain_hints:
        print(f"  └─ Domain: {', '.join(classification_result.domain_hints[:2])}")
    else:
        print(f"  └─ Domain: General")
    
    # Setup Google API
    google_key = os.getenv('GOOGLE_API_KEY')
    if not google_key:
        print("❌ No Google API key found in .env")
        return
    
    print("\n🤖 Setting up LLM providers...", end='', flush=True)
    start_time = time.time()
    
    try:
        await setup_llm_providers(google_api_key=google_key)
        print(f" ✓ ({time.time()-start_time:.1f}s)")
    except Exception as e:
        print(f" ❌ Setup failed: {e}")
        return
    
    # Run QADI phases sequentially
    print("\n🧠 Running QADI Multi-Agent Analysis...")
    print("  ├─ Question phase: Generating insightful questions")
    print("  ├─ Abduction phase: Creating hypotheses")
    print("  ├─ Deduction phase: Logical reasoning")
    print("  └─ Induction phase: Pattern synthesis")
    
    total_cost = 0.0
    all_insights = []
    
    # Phase 1: Questioning
    print("\n❓ QUESTIONING Phase...", end='', flush=True)
    phase_start = time.time()
    questions, q_cost, model_name = await run_qadi_phase("questioning", prompt, "", concrete_mode, classification_result)
    phase_time = time.time() - phase_start
    print(f" ✓ ({phase_time:.1f}s)")
    total_cost += q_cost
    
    # Show model info after first successful call
    if model_name != "unknown":
        print(f"🤖 Model: {model_name}")
    all_insights.append(f"Questions explored:\n{questions}")
    
    # Phase 2: Abduction (using previous insights)
    print("💡 ABDUCTION Phase...", end='', flush=True)
    phase_start = time.time()
    previous = f"Building on these questions:\n{questions}"
    hypotheses, h_cost, _ = await run_qadi_phase("abduction", prompt, previous, concrete_mode, classification_result)
    phase_time = time.time() - phase_start
    print(f" ✓ ({phase_time:.1f}s)")
    total_cost += h_cost
    all_insights.append(f"Hypotheses generated:\n{hypotheses}")
    
    # Phase 3: Deduction (using accumulated insights)
    print("🔍 DEDUCTION Phase...", end='', flush=True)
    phase_start = time.time()
    previous = f"Building on questions and hypotheses:\n{questions}\n{hypotheses}"
    deductions, d_cost, _ = await run_qadi_phase("deduction", prompt, previous, concrete_mode, classification_result)
    phase_time = time.time() - phase_start
    print(f" ✓ ({phase_time:.1f}s)")
    total_cost += d_cost
    all_insights.append(f"Logical deductions:\n{deductions}")
    
    # Phase 4: Induction (synthesizing all insights)
    print("🎯 INDUCTION Phase...", end='', flush=True)
    phase_start = time.time()
    previous = f"Synthesizing all insights:\n{questions}\n{hypotheses}\n{deductions}"
    patterns, i_cost, _ = await run_qadi_phase("induction", prompt, previous, concrete_mode, classification_result)
    phase_time = time.time() - phase_start
    print(f" ✓ ({phase_time:.1f}s)")
    total_cost += i_cost
    
    # Display results
    print("\n\n🔍 MULTI-AGENT QADI ANALYSIS:")
    print("=" * 70)
    
    print("\n❓ QUESTIONING:")
    for line in questions.split('\n'):
        if line.strip() and line.startswith('Q:'):
            print(f"  • {line[2:].strip()}")
    
    print("\n💡 ABDUCTION:")
    for line in hypotheses.split('\n'):
        if line.strip() and line.startswith('H:'):
            print(f"  • {line[2:].strip()}")
    
    print("\n🔍 DEDUCTION:")
    for line in deductions.split('\n'):
        if line.strip() and line.startswith('D:'):
            print(f"  • {line[2:].strip()}")
    
    print("\n🎯 INDUCTION:")
    for line in patterns.split('\n'):
        if line.strip() and line.startswith('I:'):
            print(f"  • {line[2:].strip()}")
    
    # Final synthesis
    print("\n✨ SYNTHESIS:")
    print("-" * 70)
    synthesis_prompt = f"""Based on this QADI analysis for "{prompt}":
{questions}
{hypotheses}
{deductions}
{patterns}

Provide 3 concrete, actionable recommendations that synthesize all perspectives. For each recommendation:
- Start with a clear action verb (e.g., "Implement", "Create", "Design", "Build")
- Include specific steps or methods
- Provide concrete examples when possible
- Focus on practical implementation over theoretical concepts
- Make it something someone could actually do or build

Format as:
1. **[Action-focused title]:** [Specific implementation details and examples]
2. **[Action-focused title]:** [Specific implementation details and examples]  
3. **[Action-focused title]:** [Specific implementation details and examples]"""
    
    from mad_spark_alt.core.llm_provider import llm_manager, LLMRequest
    request = LLMRequest(
        user_prompt=synthesis_prompt,
        max_tokens=400,
        temperature=0.5
    )
    
    try:
        synthesis = await asyncio.wait_for(llm_manager.generate(request), timeout=20)
        print(synthesis.content)
        total_cost += synthesis.cost
    except:
        print("(Synthesis timed out)")
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n📊 Performance Summary:")
    print(f"  ⏱️  Total time: {total_time:.1f}s")
    print(f"  💰 Total cost: {format_llm_cost(total_cost)}")
    print(f"  🤖 API calls: 5 (4 phases + synthesis)")
    if model_name != "unknown":
        print(f"  ✅ Model: {model_name}")
    else:
        print(f"  ✅ LLM mode: Multi-agent analysis")
    
    print(f"\n💡 Advantages:")
    print(f"  • Real LLM-powered insights (not templates)")
    print(f"  • Multi-perspective QADI analysis")
    print(f"  • Progressive reasoning (each phase builds on previous)")
    print(f"  • No timeout issues")
    print(f"  • Much richer than single prompt approach")

def show_help():
    """Display help information."""
    print("QADI Simple Multi-Agent - Multi-perspective AI analysis tool")
    print("=" * 60)
    print()
    print("USAGE:")
    print('  uv run python qadi_simple_multi.py "Your question"')
    print('  uv run python qadi_simple_multi.py [OPTIONS] "Your question"')
    print()
    print("DESCRIPTION:")
    print("  Analyzes questions using the QADI methodology (Question → Abduction →")
    print("  Deduction → Induction) with Google Gemini API. Provides multi-agent")
    print("  perspectives without timeouts through sequential LLM calls.")
    print()
    print("OPTIONS:")
    print("  -h, --help    Show this help message and exit")
    print("  --version     Show version information")
    print("  --concrete    Use concrete mode for practical, implementable outputs")
    print("  --type=TYPE   Manually set question type (technical, business, creative,")
    print("                research, planning, personal). Auto-detected by default.")
    print()
    print("FEATURES:")
    print("  • Real LLM-powered insights (not templates)")
    print("  • Multi-perspective QADI analysis")  
    print("  • Progressive reasoning (each phase builds on previous)")
    print("  • Adaptive prompts based on question type and complexity")
    print("  • Auto-detection of question types and domains")
    print("  • Manual question type override capability")
    print("  • No timeout issues")
    print("  • Smart cost display")
    print("  • Model identification (shows specific model used)")
    print()
    print("REQUIREMENTS:")
    print("  • Google API key in .env file (GOOGLE_API_KEY=your-key)")
    print("  • Internet connection")
    print()
    print("EXAMPLES:")
    print("  Regular mode (auto-detected question types):")
    print('    uv run python qadi_simple_multi.py "how to create AGI"')
    print('    uv run python qadi_simple_multi.py "reduce climate change"')
    print('    uv run python qadi_simple_multi.py "improve team creativity"')
    print()
    print("  Concrete mode (implementation-focused):")
    print('    uv run python qadi_simple_multi.py --concrete "build a mobile game"')
    print('    uv run python qadi_simple_multi.py --concrete "improve team productivity"')
    print('    uv run python qadi_simple_multi.py --concrete "design better user interfaces"')
    print()
    print("  Manual question type override:")
    print('    uv run python qadi_simple_multi.py --type=business "AI strategy"')
    print('    uv run python qadi_simple_multi.py --type=technical "cloud architecture"')
    print('    uv run python qadi_simple_multi.py --type=creative --concrete "logo design"')
    print()
    print("MODES:")
    print("  Regular Mode:")
    print("    • Analytical and exploratory approach")
    print("    • Focuses on deep understanding and creative insights")
    print("    • Best for brainstorming and strategic thinking")
    print()
    print("  Concrete Mode (--concrete):")
    print("    • Implementation-focused approach")
    print("    • Emphasizes practical steps and specific solutions")
    print("    • Best for project planning and actionable deliverables")
    print()
    print("QUESTION TYPES (AUTO-DETECTED):")
    print("  Technical:  Implementation, architecture, development, coding")
    print("  Business:   Strategy, growth, revenue, market, operations")
    print("  Creative:   Design, innovation, artistic, brainstorming")
    print("  Research:   Analysis, investigation, study, data, academic")
    print("  Planning:   Organization, timelines, project management")
    print("  Personal:   Individual growth, skills, habits, career")
    print()
    print("QADI METHODOLOGY:")
    print("  Question   → Generate insightful questions about the topic")
    print("  Abduction  → Create creative hypotheses and possibilities") 
    print("  Deduction  → Apply logical reasoning and derive conclusions")
    print("  Induction  → Identify patterns and synthesize insights")
    print("  Synthesis  → Combine all perspectives into actionable recommendations")
    print()
    print("OUTPUT:")
    print("  The tool provides structured analysis with:")
    print("  • Question type analysis and classification confidence")
    print("  • Detailed phase-by-phase insights")
    print("  • Final synthesis with 3 actionable recommendations")
    print("  • Performance metrics (time, cost, model used)")
    print("  • Cost information (shows 'Free' for low API costs)")
    print()
    print("PROMPT OPTIMIZATION TIPS:")
    print("  For Better Auto-Detection:")
    print("    • Use specific keywords related to your question type")
    print("    • Include domain-specific terms (e.g., 'API', 'revenue', 'design')")
    print("    • Be specific about your intent (analyze vs. build vs. plan)")
    print()
    print("  For Concrete Results:")
    print("    • Ask about building, creating, or implementing something")
    print("    • Include specific constraints (time, budget, tools)")
    print('    • Use action words: "How to build...", "Steps to create..."')
    print('    • Example: "Build a mobile game with limited budget in 3 months"')
    print()
    print("  For Analytical Results:")
    print("    • Ask about understanding, exploring, or analyzing")
    print("    • Focus on 'why' and 'what if' questions")
    print('    • Use exploratory words: "Explore...", "Understand...", "Analyze..."')
    print('    • Example: "Explore the future implications of AI on society"')
    print()
    print("  Manual Override When:")
    print("    • Auto-detection gets it wrong (low confidence)")
    print("    • You want specific expertise perspective")
    print("    • Question spans multiple types but you want focus")

def show_version():
    """Display version information."""
    print("QADI Simple Multi-Agent v1.0")
    print("Part of Mad Spark Alt - Multi-Agent Idea Generation System")
    print("Uses Google Gemini API with smart cost display and model identification")

if __name__ == "__main__":
    # Handle help and version flags
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        show_help()
    elif sys.argv[1] == "--version":
        show_version()
    else:
        # Parse command line arguments
        concrete_mode = False
        question_type_override = None
        args = sys.argv[1:]
        
        # Handle flags
        if "--concrete" in args:
            concrete_mode = True
            args.remove("--concrete")
        
        # Handle question type override
        type_flag_index = None
        for i, arg in enumerate(args):
            if arg.startswith("--type="):
                question_type_override = arg.split("=", 1)[1]
                type_flag_index = i
                break
            elif arg == "--type" and i + 1 < len(args):
                question_type_override = args[i + 1]
                type_flag_index = i
                break
        
        if type_flag_index is not None:
            # Remove type flag and its value
            if args[type_flag_index].startswith("--type="):
                args.pop(type_flag_index)
            else:
                args.pop(type_flag_index)  # Remove --type
                if type_flag_index < len(args):
                    args.pop(type_flag_index)  # Remove the value
        
        if not args:
            print("Error: Please provide a question")
            print('Usage: uv run python qadi_simple_multi.py [OPTIONS] "Your question"')
            print('Options: --concrete, --type=TYPE')
            sys.exit(1)
            
        prompt = " ".join(args)
        asyncio.run(run_simple_multi_agent_qadi(prompt, concrete_mode, question_type_override))