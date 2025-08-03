# Timeout Fix Documentation

## Problem
The QADI system was timing out at exactly 2 minutes when processing complex questions like "what is life?". The timeout message "Command timed out after 2m 0.0s" indicated a shell-level timeout, likely from the `uv run` command.

## Changes Made

### 1. **LLM Provider Timeouts** (src/mad_spark_alt/core/llm_provider.py)
- Increased OpenAI timeout: 30s → 300s (5 minutes)
- Increased Anthropic timeout: 30s → 300s (5 minutes) 
- Increased Google timeout: 60s → 300s (5 minutes)

### 2. **HTTP Request Timeout** (src/mad_spark_alt/core/retry.py)
- Increased default aiohttp timeout: 30s → 300s (5 minutes)

### 3. **Robust Orchestrator Timeouts** (src/mad_spark_alt/core/robust_orchestrator.py)
- Increased total timeout: 180s → 300s (5 minutes)
- Increased per-phase timeout: 40s → 75s
- Increased per-agent call timeout: 30s → 60s

## Running Without Shell Timeout

If you still experience the 2-minute shell timeout when using `uv run`, you have two options:

### Option 1: Use the no-timeout wrapper script
```bash
uv run python run_qadi_no_timeout.py
```

### Option 2: Run Python directly (after activating the virtual environment)
```bash
# Activate virtual environment first
source .venv/bin/activate  # or whatever your venv activation command is

# Then run directly
python examples/qadi_demo.py
```

### Option 3: Use timeout command (Unix/Linux/macOS)
```bash
# Run with 10-minute timeout
timeout 600 uv run python examples/qadi_demo.py
```

## Testing Complex Questions

With these changes, the system should now handle complex philosophical questions without timing out:

```python
# Example in qadi_demo.py
complex_questions = [
    "What is life?",
    "What is consciousness?",
    "How can we solve climate change?",
    "What is the meaning of existence?"
]
```

The system now has sufficient time to:
1. Make multiple LLM API calls (4 phases × multiple agents)
2. Handle network latency
3. Process complex reasoning tasks
4. Generate comprehensive responses