# Examples

Code examples and alternative CLI tools for Mad Spark Alt.

## Quick Start Examples

### Basic Usage (`basic_usage.py`)
Demonstrates simple QADI cycle usage with the SDK.

```bash
uv run python examples/basic_usage.py
```

Shows:
- Setting up LLM providers
- Running a basic QADI cycle
- Accessing results

---

### Complete QADI Demo (`qadi_demo.py`)
Comprehensive example showing all QADI phases in detail.

```bash
uv run python examples/qadi_demo.py
```

Shows:
- Full QADI cycle breakdown
- Phase-by-phase results
- Cost tracking

---

## Phase-Specific Examples

### Questioning Phase (`llm_questioning_demo.py`)
Demonstrates the Question phase of QADI methodology.

```bash
uv run python examples/llm_questioning_demo.py
```

---

### Abductive Reasoning (`llm_abductive_demo.py`)
Shows hypothesis generation through abductive reasoning.

```bash
uv run python examples/llm_abductive_demo.py
```

---

### Deductive Reasoning (`llm_deductive_demo.py`)
Demonstrates hypothesis evaluation and scoring.

```bash
uv run python examples/llm_deductive_demo.py
```

---

### Inductive Reasoning (`llm_inductive_demo.py`)
Shows synthesis of top hypotheses into actionable insights.

```bash
uv run python examples/llm_inductive_demo.py
```

---

## Advanced Examples

### Evolution Demo (`evolution_demo.py`)
Demonstrates genetic algorithm for idea evolution.

```bash
uv run python examples/evolution_demo.py
```

Shows:
- Evolution configuration
- Multi-generation optimization
- Fitness tracking
- Best idea selection

---

### Showcase Demo (`llm_showcase_demo.py`)
Comprehensive demonstration of all system capabilities.

```bash
uv run python examples/llm_showcase_demo.py
```

---

### User Test (`user_test.py`)
Real-world usage patterns and testing scenarios.

```bash
uv run python examples/user_test.py
```

---

## Alternative CLI Tools

These scripts provide alternative command-line interfaces to the main `msa` command, useful for specific workflows or integration scenarios.

### Simple QADI CLI (`qadi_simple_cli.py`)

Single-prompt QADI analysis without evolution.

**Usage**:
```bash
uv run python examples/qadi_simple_cli.py "your question here"
```

**Example**:
```bash
uv run python examples/qadi_simple_cli.py "How can we reduce urban traffic congestion?"
```

**Features**:
- Minimal interface for quick QADI analysis
- Outputs top hypotheses and synthesized answer
- Cost tracking included
- 203 lines of focused code

**When to use**: Quick analyses, scripts, automation pipelines

---

### Hypothesis-Driven CLI (`qadi_hypothesis_cli.py`)

QADI analysis with emphasis on hypothesis exploration.

**Usage**:
```bash
uv run python examples/qadi_hypothesis_cli.py "your question here"
```

**Example**:
```bash
uv run python examples/qadi_hypothesis_cli.py "What are potential solutions for climate change?"
```

**Features**:
- Detailed hypothesis breakdown
- Scoring transparency
- Phase-by-phase output
- 201 lines

**When to use**: Research workflows, hypothesis validation, educational purposes

---

### Multi-Perspective CLI (`qadi_multi_perspective_cli.py`)

QADI analysis from multiple analytical perspectives.

**Usage**:
```bash
uv run python examples/qadi_multi_perspective_cli.py "your question here"
```

**Example**:
```bash
uv run python examples/qadi_multi_perspective_cli.py "How should we approach AI governance?"
```

**Features**:
- Auto-detects relevant perspectives (Environmental, Personal, Technical, Business, Scientific, Philosophical)
- Parallel perspective analysis
- Perspective-weighted synthesis
- Most comprehensive analysis mode
- 301 lines

**When to use**: Complex problems requiring multiple viewpoints, strategic planning, policy analysis

---

## Comparison: Alternative CLIs vs Main `msa` Command

| Feature | `msa` | `qadi_simple_cli.py` | `qadi_hypothesis_cli.py` | `qadi_multi_perspective_cli.py` |
|---------|-------|----------------------|--------------------------|----------------------------------|
| Simple QADI | ✅ | ✅ | ✅ | ❌ |
| Multi-Perspective | ✅ | ❌ | ❌ | ✅ |
| Evolution | ✅ | ❌ | ❌ | ❌ |
| Multimodal inputs | ✅ | ❌ | ❌ | ❌ |
| PDF/URL context | ✅ | ❌ | ❌ | ❌ |
| Config options | ✅ Extensive | ❌ Basic | ❌ Basic | ❌ Basic |
| Output formats | ✅ Multiple | ❌ Terminal only | ❌ Terminal only | ❌ Terminal only |
| Best for | Production use | Quick scripts | Research | Complex analysis |

**Recommendation**: Use the main `msa` command for production workflows. Use alternative CLIs for:
- Learning the SDK API patterns
- Simple automation scripts
- Custom integration needs
- Educational purposes

---

## Running Examples

All examples require:

1. **Virtual environment setup**:
   ```bash
   uv sync
   ```

2. **API key configuration**:
   ```bash
   # Copy example .env
   cp .env.example .env

   # Add your Google API key
   echo "GOOGLE_API_KEY=your_key_here" >> .env
   ```

3. **Run examples with uv**:
   ```bash
   uv run python examples/[example_name].py
   ```

## Example Output Structure

Most examples follow this pattern:

```python
# 1. Setup
from mad_spark_alt.core import UnifiedQADIOrchestrator, setup_llm_providers

# 2. Initialize
providers = setup_llm_providers(google_api_key="...")
orchestrator = UnifiedQADIOrchestrator(providers=providers)

# 3. Execute
result = await orchestrator.run_qadi_cycle("Your question")

# 4. Access results
print(result.synthesized_answer)
for hypothesis in result.top_hypotheses:
    print(f"{hypothesis.title}: {hypothesis.score}")
```

## Next Steps

After exploring examples:

1. **Read the API documentation**: [docs/qadi_api.md](../docs/qadi_api.md)
2. **Learn CLI usage**: [docs/cli_usage.md](../docs/cli_usage.md)
3. **Understand architecture**: [ARCHITECTURE.md](../ARCHITECTURE.md)
4. **Review development guide**: [DEVELOPMENT.md](../DEVELOPMENT.md)

## Contributing Examples

When adding new examples:

1. Follow existing naming conventions (`[purpose]_demo.py` or `[feature]_cli.py`)
2. Include comprehensive docstrings
3. Add error handling and API key validation
4. Update this README with example description and usage
5. Test with `uv run python examples/your_example.py`
