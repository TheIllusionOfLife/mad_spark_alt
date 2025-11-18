# CLI Migration Guide

## Overview

Mad Spark Alt has unified its command-line interface under a single `msa` command. This guide helps you migrate from the old dual CLI system to the new unified interface.

## Quick Migration Reference

### Old Commands → New Commands

| Old Command | New Command | Notes |
|-------------|-------------|-------|
| `uv run mad_spark_alt "question"` | `msa "question"` | Default QADI analysis |
| `uv run mad_spark_alt "q" --evolve` | `msa "q" --evolve` | With evolution |
| `uv run mad-spark evolve "question"` | `msa "question" --evolve` | Evolution is now a flag |
| `uv run mad-spark list-evaluators` | `msa list-evaluators` | Unchanged |
| `uv run mad-spark evaluate "text"` | `msa --evaluate "text"` | Now uses --evaluate flag |

## Detailed Migration

### 1. Basic QADI Analysis

**Before:**
```bash
uv run mad_spark_alt "How can we reduce plastic waste?"
```

**After:**
```bash
msa "How can we reduce plastic waste?"
# Or with uv: uv run msa "How can we reduce plastic waste?"
```

### 2. QADI with Evolution

**Before:**
```bash
# Two different ways (confusing!)
uv run mad_spark_alt "question" --evolve
# OR
uv run mad-spark evolve "question"
```

**After:**
```bash
# Single unified way
msa "question" --evolve
```

### 3. Evolution with Parameters

**Before:**
```bash
uv run mad-spark evolve "question" --generations 3 --population 8
```

**After:**
```bash
msa "question" --evolve --generations 3 --population 8
```

### 4. Multimodal Analysis

**Before:**
```bash
uv run mad-spark evolve "Analyze this" --image file.png
```

**After:**
```bash
msa "Analyze this" --image file.png
# Note: Evolution requires --evolve flag now
msa "Analyze this" --image file.png --evolve
```

### 5. List Evaluators

**Before:**
```bash
uv run mad-spark list-evaluators
```

**After:**
```bash
msa list-evaluators  # Unchanged
```

### 6. Evaluate Text

**Before:**
```bash
uv run mad-spark evaluate "text" --evaluators diversity_evaluator
```

**After:**
```bash
msa --evaluate "text" --evaluate_with diversity_evaluator
# Note: Changed from subcommand to --evaluate flag, and --evaluators renamed to --evaluate_with
```

## Key Changes

### 1. Single Entry Point

- **Old**: Two commands (`mad_spark_alt` for QADI, `mad-spark` for general CLI)
- **New**: One command (`msa`) for everything
- **Benefit**: No confusion about which command to use

### 2. Default QADI Behavior

- **Old**: Required explicit subcommand (`mad-spark evolve`)
- **New**: QADI runs by default (no subcommand needed)
- **Benefit**: Simpler for the most common use case

### 3. Evolution as Flag

- **Old**: Evolution was a subcommand (`evolve`)
- **New**: Evolution is a flag (`--evolve`)
- **Benefit**: Consistent with other optional features

### 4. Shorter Alias

- **Old**: Full commands required
- **New**: Short `msa` alias available
- **Benefit**: Faster typing for frequent use

## Installation & Setup

### Reinstall Required

After updating to the new CLI, you must reinstall the package:

```bash
# Navigate to project directory
cd mad_spark_alt

# Reinstall package
uv sync
# Or: pip install -e .

# Verify new command works
msa --help
```

### Environment Variables

No changes to environment setup:

```bash
# Still uses same .env file
echo "GOOGLE_API_KEY=your_key_here" > .env
```

## Common Scenarios

### Quick Analysis

**Before:**
```bash
uv run mad_spark_alt "What are innovative ways to teach coding?"
```

**After:**
```bash
msa "What are innovative ways to teach coding?"
```

### Production Run with Evolution

**Before:**
```bash
uv run mad-spark evolve "Business innovation challenge" \
  --generations 5 \
  --population 10 \
  --temperature 1.2
```

**After:**
```bash
msa "Business innovation challenge" \
  --evolve \
  --generations 5 \
  --population 10 \
  --temperature 1.2
```

### Multimodal with Context

**Before:**
```bash
uv run mad-spark evolve "Analyze market trends" \
  --image chart.png \
  --document report.pdf \
  --url https://example.com/data
```

**After:**
```bash
msa "Analyze market trends" \
  --image chart.png \
  --document report.pdf \
  --url https://example.com/data \
  --evolve  # Add --evolve if you want evolution
```

## Script Migration

If you have scripts using the old commands, here's how to update them:

### Bash Script Example

**Before:**
```bash
#!/bin/bash
uv run mad-spark evolve "$1" --generations 3 --population 5
```

**After:**
```bash
#!/bin/bash
msa "$1" --evolve --generations 3 --population 5
```

### Python Script Example

**Before:**
```python
import subprocess
subprocess.run([
    "uv", "run", "mad-spark", "evolve",
    question,
    "--generations", "3"
])
```

**After:**
```python
import subprocess
subprocess.run([
    "msa",
    question,
    "--evolve",
    "--generations", "3"
])
```

## Troubleshooting

### Command Not Found: msa

**Problem:**
```bash
$ msa "test"
msa: command not found
```

**Solution:**
```bash
# Reinstall package
uv sync
# Or: pip install -e .

# If still not working, use full path
uv run msa "test"
```

### Old Habits

If you keep typing the old commands, create aliases:

```bash
# Add to ~/.bashrc or ~/.zshrc
alias mad_spark_alt='msa'
alias mad-spark='msa'
```

### Scripts Still Using Old Commands

**Quick Fix**: Use `uv run msa` instead of creating new aliases

```bash
# Works immediately without reinstall
uv run msa "question"
```

## FAQ

### Q: Can I still use `uv run mad_spark_alt`?

**A:** No, the old entry points are deprecated. Use `msa` or `uv run msa`.

### Q: Do I need to change my `.env` file?

**A:** No, environment setup remains the same.

### Q: What about Python API usage?

**A:** No changes to Python API. Only CLI commands changed.

### Q: Will old commands keep working?

**A:** No, old CLI entry points have been removed. You must migrate to `msa`.

### Q: How do I run evolution now?

**A:** Add `--evolve` flag: `msa "question" --evolve`

### Q: Is `evolve` still a subcommand?

**A:** No, evolution is now a flag (`--evolve`), not a subcommand.

## Benefits of Migration

1. **Simpler Mental Model**: One command, clear default behavior
2. **Less Typing**: `msa` vs `uv run mad_spark_alt`
3. **Consistent Options**: All features use flags, not subcommands
4. **Better Discovery**: `msa --help` shows everything
5. **Easier Documentation**: Single command to learn

## Need Help?

- Run `msa --help` for all options
- Check [CLI Usage Guide](cli_usage.md) for detailed examples
- See [README.md](../README.md) for quick start guide

## Summary

The new CLI is **simpler** and **more consistent**:

```bash
# Old (confusing)
uv run mad_spark_alt "q"           # For basic QADI
uv run mad-spark evolve "q"        # For evolution
uv run mad-spark list-evaluators   # For other features

# New (clear)
msa "q"                    # Basic QADI
msa "q" --evolve          # With evolution
msa list-evaluators       # Other features
```

**Bottom Line**: Replace `mad_spark_alt` and `mad-spark` with `msa` everywhere.
