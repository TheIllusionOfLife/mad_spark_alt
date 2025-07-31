# Evolution Timeout Fix Documentation

## Problem Description

When running `mad_spark_alt` with long-running commands (especially `--evolve`), commands are being killed after exactly 2 minutes (120 seconds) with the error message:
```
Command timed out after 2m 0.0s
```

This happens regardless of:
- Running Python directly: `.venv/bin/mad_spark_alt ...` (still times out)
- Using gtimeout with longer timeout: `gtimeout 600 uv run mad_spark_alt ...` (still times out)
- The Python code's calculated timeout (correctly set to 600+ seconds for evolution)

## Root Cause

The 2-minute timeout is imposed by the execution environment itself (terminal, shell, or IDE), NOT by:
- `uv run` command
- Python's asyncio timeout
- Shell timeout commands

## Working Solution: Use the nohup Wrapper Script

**Only the nohup approach successfully bypasses the timeout.**

### Usage

```bash
./run_nohup.sh "Your prompt" [options]
```

### Examples

```bash
# Evolution with custom parameters
./run_nohup.sh "Create a revolutionary game concept" --evolve --generations 3 --population 10

# Simple evolution
./run_nohup.sh "What is consciousness" --evolve

# Regular analysis with high temperature
./run_nohup.sh "Analyze this problem" --temperature 1.5
```

### How It Works

1. The script runs `mad_spark_alt` using `nohup` which detaches it from the terminal
2. Output is saved to `outputs/mad_spark_alt_output_TIMESTAMP.txt`
3. The process continues running even if the terminal is closed
4. You can monitor progress with `tail -f outputs/mad_spark_alt_output_TIMESTAMP.txt`

### Output Location

All output files are saved in the `outputs/` directory with timestamps:
- Example: `outputs/mad_spark_alt_output_20250731_190000.txt`

## Solutions That Don't Work

The following approaches were tested but **still timeout at 2 minutes**:

1. ❌ **Direct Python execution**: `.venv/bin/mad_spark_alt ...`
2. ❌ **GNU timeout command**: `gtimeout 600 uv run mad_spark_alt ...`
3. ❌ **Regular wrapper script**: `./run_evolution.sh ...` (may timeout in some environments)

## Quick Reference

### For Long-Running Tasks (Evolution, etc.)
Always use:
```bash
./run_nohup.sh "Your prompt" --evolve --generations N --population N
```

### To Monitor Progress
```bash
# Find the latest output file
ls -lt outputs/ | head -5

# Monitor the output
tail -f outputs/mad_spark_alt_output_TIMESTAMP.txt
```

### To Check if Still Running
```bash
# The script shows the PID when started
ps -p [PID]
```

## Summary

The only reliable way to run evolution or other long-running tasks is to use the `run_nohup.sh` script, which detaches the process from the terminal and avoids the 2-minute timeout imposed by the execution environment.