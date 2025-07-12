# User Testing Guide for Mad Spark Alt

Welcome! This guide will help you test the Mad Spark Alt idea generation system with your own prompts.

## Quick Start

### Option 1: Interactive Test Script (Recommended)

```bash
# Run the interactive test
uv run python examples/user_test.py

# Or run with a specific problem directly
uv run python examples/user_test.py "How can we make cities more sustainable?"

# Or run in continuous interactive mode
uv run python examples/user_test.py --interactive
```

### Option 2: Evolution Demo with Custom Prompt

```bash
# See how ideas evolve through genetic algorithms
uv run python examples/evolution_demo.py
```

### Option 3: Command Line Interface

```bash
# Evaluate a single idea
uv run mad-spark evaluate "Your creative text or idea here"

# List available evaluators
uv run mad-spark list-evaluators
```

## What You'll Experience

### 1. **QADI Cycle** (Question → Abduction → Deduction → Induction)
The system will take your problem statement through four thinking phases:
- **❓ Questioning**: Generates diverse questions about your problem
- **💡 Abduction**: Creates hypotheses and creative leaps
- **🔍 Deduction**: Validates ideas with logical reasoning
- **🔗 Induction**: Synthesizes patterns and insights

### 2. **Smart Agent System**
- **With API Keys**: 🤖 AI-powered agents provide sophisticated reasoning
- **Without API Keys**: 📝 Template agents provide structured generation

### 3. **Evolution System** (Optional)
- Ideas can be evolved through genetic algorithms
- Multiple generations improve idea quality
- Fitness scoring based on creativity metrics

## Setting Up API Keys (Optional but Recommended)

For the best experience with AI-powered generation:

```bash
# Choose one or more providers:
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  
export GOOGLE_API_KEY="your-google-key"
```

## Example Prompts to Try

### Business & Innovation
- "How can small businesses compete with large corporations in the digital age?"
- "What innovative business models could address climate change?"
- "How might we revolutionize remote work collaboration?"

### Technology & Society
- "How can AI improve healthcare accessibility in rural areas?"
- "What are creative solutions to digital privacy concerns?"
- "How might we bridge the digital divide in education?"

### Environmental & Sustainability
- "How can cities become carbon-neutral by 2030?"
- "What innovative approaches could solve ocean plastic pollution?"
- "How might vertical farming transform urban food systems?"

### Creative & Abstract
- "What if gravity worked differently on weekends?"
- "How would society change if we could share dreams?"
- "Design a new sport for zero-gravity environments"

## Interactive Mode Commands

When using `user_test.py --interactive`:

- **Basic Input**: Just type your problem statement
- **With Context**: `Your problem context: Additional context here`
- **Set Idea Count**: `Your problem ideas:5` (generates 5 ideas per phase)
- **Help**: Type `help` for options
- **Quit**: Type `quit` to exit

## Understanding the Output

### Phase Results
Each phase shows:
- **Agent Type**: 🤖 (LLM) or 📝 (Template)
- **Generated Ideas**: The actual ideas produced
- **Reasoning**: For LLM agents, you'll see reasoning behind ideas
- **Confidence**: LLM agents provide confidence scores

### Cost Tracking
If using LLM agents, you'll see:
- **Execution Time**: How long the generation took
- **LLM Cost**: Approximate cost in USD for API calls

### Synthesized Ideas
All ideas from all phases are collected and displayed, grouped by their originating phase.

## Tips for Best Results

1. **Be Specific**: "How can we reduce food waste in university cafeterias?" works better than "How to save food?"

2. **Add Context**: Use the context option to guide the generation:
   ```
   problem: "How to improve public transport?"
   context: "Focus on accessibility for elderly and disabled users"
   ```

3. **Experiment with Different Domains**: The system works across technical, creative, business, and social domains

4. **Try Evolution**: After generating ideas, try the evolution demo to see how they improve over generations

## Troubleshooting

### No Ideas Generated?
- Check if you have API keys set (for better results)
- Try a clearer problem statement
- Ensure you've installed dependencies: `uv sync`

### Import Errors?
```bash
# Make sure the package is installed
uv pip install -e .
```

### Want More Ideas?
Use the `ideas:N` command in interactive mode or modify the config in the script.

## Next Steps

1. **Test Different Problems**: Try various domains and complexity levels
2. **Compare Modes**: Test with and without API keys to see the difference
3. **Try Evolution**: Run `evolution_demo.py` to see genetic algorithm enhancement
4. **Explore the API**: Check `examples/basic_usage.py` for programmatic usage

## Feedback

After testing, consider:
- Which types of problems worked best?
- Was the output helpful and creative?
- How was the user experience?
- What features would you like to see?

Happy testing! 🚀