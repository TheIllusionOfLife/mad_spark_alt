#!/bin/bash
# Script to run mad_spark_alt commands using nohup to avoid terminal timeout issues
# This detaches the process from the terminal completely, avoiding 2-minute timeout issues

# Check if at least one argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 \"your prompt\" [options...]"
    echo ""
    echo "Examples:"
    echo "  $0 \"Create a game concept\" --evolve --generations 3 --population 10"
    echo "  $0 \"What is consciousness\" --evolve"
    echo "  $0 \"Analyze this problem\" --temperature 1.5"
    echo ""
    echo "This script runs mad_spark_alt in the background to avoid 2-minute timeout issues."
    exit 1
fi

# Ensure we're in the right directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found at .venv"
    echo "Please run 'uv sync' first"
    exit 1
fi

# Check if mad_spark_alt is installed
if [ ! -f ".venv/bin/mad_spark_alt" ]; then
    echo "Error: mad_spark_alt not found in virtual environment"
    echo "Please run 'uv pip install -e .' first"
    exit 1
fi

# Ensure outputs directory exists
mkdir -p outputs

# Create output filename with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="outputs/mad_spark_alt_output_${TIMESTAMP}.txt"

echo "Starting Mad Spark Alt (detached to avoid timeout)..."
echo "Command: .venv/bin/mad_spark_alt $@"
echo "Output will be saved to: $OUTPUT_FILE"
echo "----------------------------------------"

# Run with nohup, redirecting output to file
nohup .venv/bin/mad_spark_alt "$@" > "$OUTPUT_FILE" 2>&1 &
PID=$!

echo "Process started with PID: $PID"
echo ""
echo "To monitor progress:"
echo "  tail -f $OUTPUT_FILE"
echo ""
echo "To check if still running:"
echo "  ps -p $PID"
echo ""
echo "The process will continue even if you close this terminal."