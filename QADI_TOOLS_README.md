# QADI Tools - Command-Line Interface for Mad Spark Alt

This directory contains various QADI (Question → Abduction → Deduction → Induction) command-line tools that demonstrate different aspects of the Mad Spark Alt system.

## 🎯 Available Tools

### 1. `qadi.py` - Fast Single-Prompt QADI
**Usage**: `uv run python qadi.py "Your question"`

**What it does**: 
- Single LLM call with structured QADI prompt
- Fast execution (7-12 seconds)
- Simple 3-answer format
- Good for quick analysis

**Example**:
```bash
uv run python qadi.py "how to improve team productivity"
```

**Pros**: Fast, simple, low cost
**Cons**: Limited depth, essentially a formatted prompt wrapper

---

### 2. `qadi_multi_agent.py` - Full Multi-Agent QADI ⭐
**Usage**: `uv run python qadi_multi_agent.py "Your question" [--evolve]`

**What it does**:
- Uses the complete Mad Spark Alt multi-agent system
- 4 specialized thinking agents (Question, Abduction, Deduction, Induction)
- Intelligent agent selection (LLM vs template fallback)
- Rich reasoning trails and metadata
- Optional genetic evolution with `--evolve`

**Example**:
```bash
uv run python qadi_multi_agent.py "how can AI improve healthcare"
uv run python qadi_multi_agent.py "sustainable energy solutions" --evolve
```

**Pros**: Deep analysis, specialized reasoning, extensible, rich output
**Cons**: Slower (15-30s), higher cost

---

### 3. `compare_qadi_approaches.py` - Side-by-Side Comparison
**Usage**: `uv run python compare_qadi_approaches.py "Your question"`

**What it does**:
- Runs both simple prompt and multi-agent approaches
- Provides detailed performance comparison
- Shows time, cost, and quality differences
- Demonstrates why multi-agent is superior

**Example**:
```bash
uv run python compare_qadi_approaches.py "improve online learning"
```

**Output includes**:
- Execution time comparison
- Cost analysis
- Quality and depth differences
- Agent type usage
- Concrete examples of each approach

---

### 4. `qadi_evolution_demo.py` - Genetic Evolution Showcase 🧬
**Usage**: `uv run python qadi_evolution_demo.py "Your question"`

**What it does**:
- Full QADI multi-agent idea generation
- Genetic evolution of generated ideas
- Fitness-based selection and optimization
- Population diversity management
- Demonstrates advanced capabilities

**Example**:
```bash
uv run python qadi_evolution_demo.py "reduce food waste in restaurants"
```

**Shows**:
- Multi-generation evolution
- Fitness improvement metrics
- Population diversity tracking
- Best evolved ideas
- Why this beats simple prompting

---

## 🚀 Quick Start

1. **Setup environment**:
```bash
# Install dependencies
uv sync

# Set up API keys in .env
echo "GOOGLE_API_KEY=your_key_here" >> .env
```

2. **Try the simple approach**:
```bash
uv run python qadi.py "how to reduce stress"
```

3. **Experience the full system**:
```bash
uv run python qadi_multi_agent.py "how to reduce stress"
```

4. **Compare approaches**:
```bash
uv run python compare_qadi_approaches.py "how to reduce stress"
```

5. **See genetic evolution in action**:
```bash
uv run python qadi_evolution_demo.py "how to reduce stress"
```

## 🎯 When to Use Which Tool

### Use `qadi.py` when:
- You need quick answers
- Cost is a major concern
- Simple 3-answer format is sufficient
- You're doing rapid iteration

### Use `qadi_multi_agent.py` when:
- You need deep, multi-perspective analysis
- Quality is more important than speed
- You want rich reasoning trails
- You're solving complex problems
- You want to leverage the full Mad Spark Alt capabilities

### Use `compare_qadi_approaches.py` when:
- You want to understand the differences
- You're evaluating the system
- You need to justify the multi-agent approach
- You're demonstrating capabilities

### Use `qadi_evolution_demo.py` when:
- You want to see cutting-edge capabilities
- You're working on optimization problems
- You need the absolute best solutions
- You're showcasing advanced AI techniques

## 📊 Performance Characteristics

| Tool | Speed | Cost | Depth | Agents | Evolution |
|------|-------|------|-------|--------|-----------|
| `qadi.py` | Fast (7-12s) | Low | Basic | 1 | No |
| `qadi_multi_agent.py` | Medium (15-30s) | Medium | High | 4+ | Optional |
| `compare_qadi_approaches.py` | Slow (both) | Medium | Comparison | Both | No |
| `qadi_evolution_demo.py` | Slow (30-60s) | High | Highest | 4+ | Yes |

## 🔍 Key Differences from Simple Prompting

The multi-agent tools provide significant advantages over simple prompting:

### 🧠 **Intellectual Rigor**
- **Simple Prompt**: Single perspective, limited by one LLM call
- **Multi-Agent**: Four specialized thinking perspectives, each optimized for its cognitive approach

### 🔄 **Progressive Reasoning**
- **Simple Prompt**: Static analysis, no building on previous insights
- **Multi-Agent**: Each phase builds on previous phases, creating deeper understanding

### 🎯 **Specialized Expertise**
- **Simple Prompt**: Generic analysis approach
- **Multi-Agent**: Question agents focus on inquiry, Abduction on creativity, Deduction on logic, Induction on patterns

### 🧬 **Evolution & Optimization**
- **Simple Prompt**: Fixed output, no improvement
- **Multi-Agent**: Genetic algorithms can evolve ideas across generations

### 📊 **Extensibility**
- **Simple Prompt**: Hard to extend or customize
- **Multi-Agent**: Pluggable agents, custom evaluators, configurable strategies

### 🛡️ **Robustness**
- **Simple Prompt**: Single point of failure
- **Multi-Agent**: Fallback mechanisms, error recovery, multiple API providers

## 🎓 Educational Value

These tools demonstrate key AI concepts:

1. **Multi-Agent Systems**: Coordination between specialized AI agents
2. **Genetic Algorithms**: Evolution of solutions through selection and mutation
3. **Fitness Evaluation**: Quantitative assessment of solution quality
4. **Progressive Reasoning**: Building complex analysis through sequential phases
5. **Fallback Strategies**: Graceful degradation when preferred methods fail

## 🔧 Configuration

All tools support environment-based configuration:

```bash
# Required
GOOGLE_API_KEY=your_google_key

# Optional (for multi-provider support)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

## 🐛 Troubleshooting

**Common Issues**:

1. **API Key Missing**: Ensure `.env` file has valid API keys
2. **Timeout Errors**: Google API can be slow; tools have 30s timeout
3. **Cost Concerns**: Multi-agent tools use more tokens; monitor usage
4. **Import Errors**: Run `uv sync` to ensure dependencies are installed

**Debug Mode**:
Add debug logging by setting environment variable:
```bash
export LOG_LEVEL=DEBUG
uv run python qadi_multi_agent.py "your question"
```

## 🎯 Conclusion

These tools showcase the evolution from simple prompt engineering to sophisticated multi-agent AI systems. While `qadi.py` provides a fast baseline, the multi-agent tools demonstrate the true potential of the Mad Spark Alt framework for generating high-quality, evolved solutions to complex problems.

The comparison and evolution demos clearly show why Mad Spark Alt is much more than just a prompt wrapper - it's a comprehensive framework for AI-powered collaborative thinking and solution optimization.