# QADI Tools - Command-Line Interface for Mad Spark Alt

This directory contains various QADI (Question → Abduction → Deduction → Induction) command-line tools that demonstrate different aspects of the Mad Spark Alt system.

## 🎯 Available Tools

### 1. `qadi_simple_multi.py` - Multi-Agent QADI (Recommended) ⭐
**Usage**: `uv run python qadi_simple_multi.py "Your question"`

**What it does**:
- Uses Google API with multiple simple LLM calls
- 4 specialized thinking phases (Question, Abduction, Deduction, Induction)
- Progressive reasoning - each phase builds on previous insights
- Final synthesis combining all perspectives
- No timeout issues

**Example**:
```bash
uv run python qadi_simple_multi.py "how to create AGI"
uv run python qadi_simple_multi.py "improve team productivity"
```

**Pros**: Real LLM insights, multi-perspective analysis, reliable execution
**Cons**: Multiple API calls (higher cost than single prompt)

---

### 2. `qadi.py` - Fast Single-Prompt QADI
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

### 3. `qadi_multi_agent_fixed.py` - LLM Multi-Agent (Experimental) ⚠️
**Usage**: `uv run python qadi_multi_agent_fixed.py "Your question" [--evolve]`

**What it does**:
- Advanced version with LLM-powered agents
- Comprehensive timeout handling and fallbacks
- Rich LLM-generated reasoning
- Optional genetic evolution with `--evolve`
- **Warning**: May still timeout on complex questions

**Example**:
```bash
uv run python qadi_multi_agent_fixed.py "simple question here"
```

**Pros**: LLM-powered insights, rich reasoning, genetic evolution
**Cons**: May timeout, slower, higher cost, experimental

---

### 4. `compare_qadi_approaches.py` - Side-by-Side Comparison
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

### 5. `qadi_working.py` - Template Agent Demo (NOT RECOMMENDED) ❌
**Usage**: `uv run python qadi_working.py "Your question"`

**What it does**:
- Uses template agents that produce generic, meaningless responses
- Does NOT use any LLM or Google API
- Only useful for understanding the multi-agent architecture

**Why NOT to use**:
- Template responses don't engage with your actual question
- No real insights or analysis
- Produces the same generic output for any input

---

### 6. `qadi_evolution_demo.py` - Genetic Evolution Showcase 🧬
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

### Use `qadi_simple_multi.py` when: ⭐ RECOMMENDED
- You want real multi-agent analysis with Google API
- You need multiple perspectives on complex problems
- You want progressive reasoning that builds insights
- You need reliable execution without timeouts
- Quality is more important than cost

### Use `qadi.py` when:
- You need quick answers
- Cost is a major concern
- Simple 3-answer format is sufficient
- You're doing rapid iteration

### Use `qadi_multi_agent_fixed.py` when:
- You want to experiment with advanced LLM agents
- You're willing to risk timeouts
- You're debugging the full system

### NEVER use `qadi_working.py`:
- Template agents are meaningless
- Does not use Google API or any LLM
- Produces generic responses regardless of input

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

| Tool | Speed | Cost | Depth | Uses LLM | Reliability | Recommended |
|------|-------|------|-------|----------|-------------|-------------|
| `qadi_simple_multi.py` | Medium (20-30s) | Medium | High | ✅ Google API | ✅ High | ⭐ YES |
| `qadi.py` | Fast (7-12s) | Low | Basic | ✅ Google API | ✅ High | For quick use |
| `qadi_multi_agent_fixed.py` | Slow (timeout risk) | High | Highest | ✅ Multiple | ⚠️ Experimental | No |
| `qadi_working.py` | Fast (<1s) | None | Meaningless | ❌ Templates | ✅ High | ❌ NEVER |
| `qadi_evolution_demo.py` | Slow (30-60s) | High | Highest | ✅ Multiple | ⚠️ Experimental | No |

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

### Timeout Issues ⚠️

**Problem**: Some LLM-powered multi-agent tools may timeout
**Root Cause**: Complex LLM agent coordination can hang during execution
**Solution**: Use reliable alternatives

**Diagnostic Tools**:
```bash
# Test all components systematically
uv run python diagnose_timeout.py

# Find exact failure point
uv run python minimal_qadi_test.py
```

**Immediate Solutions**:
```bash
# ✅ BEST: Multi-agent with Google API (recommended)
uv run python qadi_simple_multi.py "your question"

# ✅ Simple fast approach
uv run python qadi.py "your question"

# ❌ NEVER use template agents
# uv run python qadi_working.py  # DON'T USE THIS

# ⚠️ Experimental LLM version (may timeout)
uv run python qadi_multi_agent_fixed.py "your question"
```

### Other Common Issues

1. **API Key Missing**: Ensure `.env` file has valid API keys
2. **Cost Concerns**: Multi-agent tools use more tokens; monitor usage
3. **Import Errors**: Run `uv sync` to ensure dependencies are installed

**Debug Mode**:
```bash
export LOG_LEVEL=DEBUG
uv run python qadi_simple_multi.py "your question"
```

**For detailed timeout analysis, see**: `TIMEOUT_ANALYSIS.md`

## 🎯 Conclusion

These tools showcase the evolution from simple prompt engineering to sophisticated multi-agent AI systems. While `qadi.py` provides a fast baseline, **`qadi_simple_multi.py` demonstrates true multi-agent benefits using Google API** with reliable execution and real LLM-powered insights.

## ⚠️ IMPORTANT: Always Use Google API

**Template agents are meaningless** - they produce generic responses that don't engage with your actual question. Always use tools that leverage Google API or other LLMs for real insights:

✅ **USE**: `qadi_simple_multi.py` (multi-agent with Google API)
✅ **USE**: `qadi.py` (simple but uses Google API)
❌ **NEVER USE**: `qadi_working.py` (template agents, no LLM)

Mad Spark Alt successfully demonstrates that it's much more than just a prompt wrapper - it's a comprehensive framework for AI-powered collaborative thinking that **requires real LLM integration to provide value**.