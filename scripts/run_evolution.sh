#!/bin/bash
set -euo pipefail
# Script to run msa (Mad Spark Alt) with evolution
# This script runs the command in the foreground, replacing the shell process.

# Check if at least one argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 \"your prompt\" [--evolve] [--generations N] [--population N] [other options...]"
    echo "Example: $0 \"Create a game concept\" --evolve --generations 3 --population 10"
    exit 1
fi

# Ensure we're in the project root directory (one level up from scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.." || {
    echo "Error: Failed to change to project directory" >&2
    exit 1
}

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found at .venv"
    echo "Please run 'uv sync' first"
    exit 1
fi

# Check if msa is installed
if [ ! -f ".venv/bin/msa" ]; then
    echo "Error: msa not found in virtual environment"
    echo "Please run 'uv pip install -e .' first"
    exit 1
fi

echo "Starting Mad Spark Alt with evolution..."
echo "Command: .venv/bin/msa $@"
echo ""
echo "⚠️  WARNING: This may timeout after 2 minutes in some environments."
echo "   If you experience timeout issues, use scripts/run_nohup.sh instead."
echo "----------------------------------------"

# Run the command directly (not in background) to see output
# The key is using the venv Python directly, which avoids uv run's timeout
exec .venv/bin/msa "$@"