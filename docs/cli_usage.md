# CLI Usage Guide

## Overview

Mad Spark Alt provides a unified command-line interface (`msa`) for QADI analysis, idea evolution, and creativity evaluation. This guide covers all available commands and usage patterns.

## Installation and Setup

```bash
# Install the package
pip install -e .

# Or with uv (recommended)
uv sync

# Verify installation
msa --help
# Or with uv: uv run msa --help
```

## Environment Setup

### Required Environment Variables

For QADI multi-agent analysis and LLM functionality, set up API keys:

```bash
# Google Gemini API (REQUIRED - primary LLM provider)
export GOOGLE_API_KEY="your-google-api-key"

# Optional: OpenAI API (alternative LLM provider)
export OPENAI_API_KEY="your-openai-api-key"

# Optional: Anthropic Claude API (alternative LLM provider)
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Alternative: Create .env file in project root
echo "GOOGLE_API_KEY=your-key-here" > .env
```

**Note**: At least one LLM API key is required for the QADI system to function. Google API is recommended for best results and lowest cost.

## Basic Usage

### Default Command: QADI Analysis

The `msa` command runs QADI analysis by default (no subcommand needed):

```bash
# Basic QADI analysis
msa "How can we reduce plastic waste?"

# With verbose output
msa "Your question" --verbose

# Display help
msa --help
```

### Global Options

```bash
msa --help                    # Show help
msa --verbose                 # Enable verbose output
msa --version                 # Show version
```

## QADI Analysis Options

### Basic QADI Usage

```bash
# Simple question
msa "How can we improve urban sustainability?"

# With custom temperature (creativity level)
msa "Your question" --temperature 1.2

# With context
msa "Improve customer service" --context "E-commerce platform"
```

### Evolution Mode

Add `--evolve` to use genetic algorithms for idea optimization:

```bash
# Basic evolution
msa "How can we reduce food waste?" --evolve

# Custom evolution parameters (generations = 2, population = 5 by default)
msa "Climate solutions" --evolve --generations 3 --population 8

# Use traditional operators for faster evolution
msa "Business strategies" --evolve --traditional

# With semantic diversity (slower but more accurate)
msa "Innovation challenge" --evolve --diversity-method semantic
```

### Multimodal Analysis

Analyze images, documents, and URLs alongside text:

```bash
# Analyze an image
msa "Analyze this design for improvement" --image design.png

# Process a PDF document
msa "Summarize key findings" --document research.pdf

# Combine multiple modalities
msa "Compare these approaches" \
  --image chart1.png \
  --image chart2.png \
  --url https://example.com/article

# Multiple documents and URLs
msa "Synthesize insights" \
  --document report1.pdf \
  --document report2.pdf \
  --url https://source1.com \
  --url https://source2.com
```

## Advanced QADI Options

### Complete Option List

```bash
msa "Your question here" \
  --temperature 1.5 \              # Creativity level 0.0-2.0 (default: 0.8)
  --evolve \                       # Enable genetic evolution
  --generations 3 \                # Evolution generations (default: 2)
  --population 8 \                 # Population size (default: 5)
  --traditional \                  # Use traditional operators (faster)
  --diversity-method semantic \    # Use semantic diversity (slower, more accurate)
  --context "Additional context" \ # Extra context for analysis
  --verbose                        # Detailed output
```

### Temperature Control

The `--temperature` flag controls creativity in hypothesis generation:
- `0.0-0.5`: Conservative, practical ideas
- `0.6-1.0`: Balanced creativity (default: 0.8)
- `1.1-2.0`: Highly creative, unconventional ideas

### Evolution Process

When `--evolve` is enabled:

1. **QADI Analysis**: Generates initial hypotheses
2. **Fitness Evaluation**: Scores on 5 criteria (novelty, impact, cost, feasibility, risks)
3. **Selection**: Best ideas selected for breeding
4. **Crossover**: Combines elements from parent ideas
5. **Mutation**: Introduces variations
6. **Repeat**: Continues for specified generations

## Subcommands

While QADI analysis is the default, additional commands are available:

### List Evaluators

```bash
# Show all available evaluators
msa list-evaluators
```

### Evaluate Text

```bash
# Evaluate creativity of text
msa evaluate "The quantum cat leaped through dimensions"

# With specific evaluators
msa evaluate "text" --evaluators diversity_evaluator

# With multiple evaluators
msa evaluate "text" --evaluators diversity_evaluator,quality_evaluator

# Use all evaluators (default)
msa evaluate "text"
```

## Example Outputs

### QADI Analysis Output

```
## 💡 Phase 1: Core Question Analysis (Questioning)

The refined question is: "What specific, actionable strategies can reduce plastic
waste in oceans while considering environmental, economic, and social factors?"

## 💡 Phase 2: Hypothesis Generation (Abduction)

1. Autonomous ocean drones with ML-powered plastic detection.
   Deploy AI-guided vessels that identify and collect plastic...

2. Blockchain-tracked plastic credits incentivizing cleanup.
   Create a digital marketplace where coastal communities...

3. Bioengineered bacteria that safely decompose ocean plastic.
   Develop specialized microorganisms that break down...

## 💡 Phase 3: Evaluation & Best Answer (Deduction)

Best Hypothesis: Autonomous ocean drones with ML-powered plastic detection
- Impact: 0.92
- Feasibility: 0.78
- Accessibility: 0.85

## 💡 Phase 4: Real-World Examples (Induction)

1. The Ocean Cleanup project in the Pacific Garbage Patch...
2. Indonesia's coastal community plastic credit program...
3. Research at MIT on plastic-eating enzymes...

💰 LLM Cost: $0.0234
```

### Evolution Output

```
🧬 Evolution Results

✅ Generated 3 initial hypotheses
✅ Evolution completed in 45.2s

🏆 Top Evolved Ideas:
1. Autonomous ocean drones with ML-powered plastic detection... (Fitness: 0.892)
2. Blockchain-tracked plastic credits incentivizing cleanup... (Fitness: 0.847)
3. Bioengineered bacteria that safely decompose ocean plastic... (Fitness: 0.823)

📊 Evolution Stats:
• Fitness improvement: 47.3%
• Ideas evaluated: 60
• Best from generation: 4

💰 Total LLM Cost: $0.0156
```

## Configuration

### Environment Variables

```bash
# Custom config directory
export MAD_SPARK_CONFIG_DIR="~/.config/mad_spark_alt"

# Enable detailed logging
export MAD_SPARK_LOG_LEVEL="DEBUG"
```

## Cost Information

### Typical Costs (Gemini 2.5 Flash)

| Configuration | Time | Cost | Quality |
|--------------|------|------|---------|
| Basic QADI only | ~10s | $0.002 | Good baseline |
| Evolution (pop=3, gen=2) | ~60s | $0.005 | Better diversity |
| Evolution (pop=5, gen=3) | ~180s | $0.008 | Great results |
| Evolution (pop=10, gen=5) | ~450s | $0.016 | Maximum quality |
| With semantic diversity | +30s | +$0.001 | Conceptual diversity |

**Note**: Actual costs may vary based on prompt length and response verbosity.

## Troubleshooting

### Common Issues

```bash
# Check API key setup
msa "test" --verbose

# If msa command not found, use full path
uv run msa "test"

# For long-running tasks that timeout, use nohup wrapper
./run_nohup.sh "Your question" --evolve
```

### 2-Minute Timeout in Some Environments

When running long commands (especially with `--evolve`), you may encounter a timeout after exactly 2 minutes. This is caused by the execution environment (terminal/shell/IDE), not the application.

**Solution**: Use the nohup wrapper script:

```bash
# Instead of: msa "prompt" --evolve
# Use: ./run_nohup.sh "prompt" --evolve

./run_nohup.sh "Create a game concept" --evolve --generations 3 --population 10
```

Output will be saved to `outputs/msa_output_TIMESTAMP.txt`.

## Examples and Use Cases

### Academic Research

```bash
# Generate research questions
msa "How can AI improve scientific discovery?" \
  --context "Academic research context" \
  --temperature 1.2
```

### Creative Writing

```bash
# Generate story ideas
msa "Create an engaging science fiction story" \
  --temperature 1.5 \
  --evolve
```

### Business Innovation

```bash
# Generate business solutions
msa "How can we improve customer retention?" \
  --context "SaaS business model" \
  --evolve \
  --generations 3
```

### Environmental Planning

```bash
# Analyze with images and data
msa "Assess this urban development plan" \
  --image site_plan.png \
  --document environmental_impact.pdf \
  --url https://city-planning-docs.com
```

This guide covers the essential usage of the `msa` command for QADI analysis, evolution, and multimodal processing.
