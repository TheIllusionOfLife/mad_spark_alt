# Scripts

Utility scripts for running Mad Spark Alt in different execution environments.

## Available Scripts

### `run_nohup.sh`

Background execution wrapper that avoids terminal timeout issues.

**Purpose**: Some terminal environments (IDEs, SSH sessions) impose 2-minute timeouts on foreground processes. This script uses `nohup` to run Mad Spark Alt in the background, allowing long-running tasks to complete without interruption.

**Usage**:
```bash
scripts/run_nohup.sh "your prompt" [options...]
```

**Example**:
```bash
scripts/run_nohup.sh "Create a game concept" --evolve --generations 3 --population 10
```

**Features**:
- Runs in background, immune to terminal timeouts
- Creates timestamped output files in `outputs/`
- Output format: `outputs/msa_output_YYYYMMDD_HHMMSS.txt`
- Monitors process completion
- Shows full output when complete

**When to use**: Long-running tasks (evolution with multiple generations, large populations, or complex prompts)

---

### `run_evolution.sh`

Foreground execution wrapper for direct command invocation.

**Purpose**: Runs Mad Spark Alt in foreground mode, replacing the shell process. Useful when you want to see live output and don't expect timeout issues.

**Usage**:
```bash
scripts/run_evolution.sh "your prompt" [options...]
```

**Example**:
```bash
scripts/run_evolution.sh "Create a game concept" --evolve --generations 2 --population 5
```

**Features**:
- Foreground execution with live output
- Uses virtual environment Python directly (bypasses `uv run` timeout)
- Provides immediate feedback

**When to use**: Quick tests, debugging, or environments without timeout constraints

**Note**: If you experience timeout issues, switch to `run_nohup.sh` instead.

---

## Script Comparison

| Feature | `run_nohup.sh` | `run_evolution.sh` |
|---------|----------------|-------------------|
| Execution mode | Background | Foreground |
| Output destination | File (`outputs/`) | Terminal |
| Timeout resistance | ✅ Immune | ⚠️ May timeout |
| Live output | ❌ No | ✅ Yes |
| Process monitoring | ✅ Yes | ❌ N/A |
| Best for | Long-running tasks | Quick tests, debugging |

## Troubleshooting

### "Command not found" errors
Both scripts automatically navigate to the project root directory. Run them from anywhere in the project:
```bash
# From project root
./scripts/run_nohup.sh "prompt"

# From any subdirectory
../scripts/run_nohup.sh "prompt"
```

### Virtual environment issues
If you see "Virtual environment not found" or "msa not found":
```bash
# Install dependencies
uv sync

# Install the package in development mode
uv pip install -e .
```

### Permission denied
Make scripts executable:
```bash
chmod +x scripts/run_nohup.sh scripts/run_evolution.sh
```

## Technical Details

**Directory Navigation**: Both scripts automatically change to the project root directory (one level up from `scripts/`), ensuring virtual environment and outputs directory are accessible regardless of where you invoke them.

**Command Used**: Both scripts use `.venv/bin/msa` directly instead of `uv run msa` to avoid `uv`'s process management overhead that can contribute to timeout issues.

**Exit Handling**: Scripts use `exec` (foreground) or proper background job management (nohup) to ensure clean process lifecycle.
