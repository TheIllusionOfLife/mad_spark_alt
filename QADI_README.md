# QADI Command Line Tool

A command-line tool that uses Google's Gemini API to analyze questions using the QADI (Question, Abduction, Deduction, Induction) methodology.

## Setup

1. Add your Google API key to `.env`:
```bash
GOOGLE_API_KEY=your-google-api-key-here
```

2. Ensure dependencies are installed:
```bash
uv sync
```

## Usage

```bash
uv run python qadi.py "Your question here"
```

## Examples

```bash
# Personal growth
uv run python qadi.py "how to live longer"

# Practical advice
uv run python qadi.py "what are 3 ways to reduce stress"

# Science questions
uv run python qadi.py "Why is the sky blue?"

# Creative prompts
uv run python qadi.py "Unique mobile game concept"

# Travel planning
uv run python qadi.py "Suggest 5 fun ways to spend the weekend in Tokyo"
```

## How it Works

The tool uses a single optimized LLM call to:
1. Analyze your question using QADI methodology
2. Generate a key question, hypothesis, and logical deduction
3. Extract 3 practical answers based on the analysis

Typical response time: 7-12 seconds

## Output Format

```
🧠 QADI ANALYSIS:
- Question: A deeper question to explore
- Hypothesis: A creative theory or assumption  
- Deduction: A logical conclusion

✅ PRACTICAL ANSWERS:
1. Answer based on the question exploration
2. Answer based on the hypothesis
3. Answer based on the logical deduction
```