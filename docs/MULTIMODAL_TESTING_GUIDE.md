# Multimodal Support Testing Guide

This document provides comprehensive testing procedures for the multimodal support feature (Phase 3).

## Prerequisites

1. **API Key Setup**
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

2. **Test Data Preparation**
   ```bash
   # Create test data directory
   mkdir -p test_multimodal_data
   cd test_multimodal_data

   # Create a simple test image (requires PIL)
   python3 << 'EOF'
   from PIL import Image, ImageDraw

   img = Image.new('RGB', (400, 300), color='white')
   draw = ImageDraw.Draw(img)

   draw.rectangle([50, 50, 350, 250], outline='black', width=2)
   draw.text((70, 70), "Financial Chart", fill='black')
   draw.text((70, 100), "Revenue: $100K", fill='blue')
   draw.text((70, 130), "Costs: $60K", fill='red')
   draw.text((70, 160), "Profit: $40K", fill='green')

   draw.rectangle([70, 190, 170, 230], fill='blue')
   draw.rectangle([180, 200, 250, 230], fill='red')
   draw.rectangle([260, 180, 330, 230], fill='green')

   img.save('test_chart.png')
   print("Created test_chart.png")
   EOF

   cd ..
   ```

## Automated Tests

### 1. Unit Tests (No API Key Required)

```bash
# Run all unit tests
uv run pytest tests/test_cli_multimodal.py -v

# Run orchestrator signature tests
uv run pytest tests/core/test_orchestrator_multimodal_signatures.py -v

# Run PhaseInput tests
uv run pytest tests/core/test_phase_input_multimodal.py -v
```

Expected: All tests pass without requiring API key.

### 2. Integration Tests (API Key Required)

```bash
# Run all integration tests with real API
GOOGLE_API_KEY=xxx uv run pytest tests/test_real_api_multimodal.py -v

# Run specific test classes
GOOGLE_API_KEY=xxx uv run pytest tests/test_real_api_multimodal.py::TestRealAPIImageProcessing -v
GOOGLE_API_KEY=xxx uv run pytest tests/test_real_api_multimodal.py::TestRealAPIURLProcessing -v
GOOGLE_API_KEY=xxx uv run pytest tests/test_real_api_multimodal.py::TestRealAPICombinedMultimodal -v
```

Expected Results:
- All tests pass
- Images are processed correctly
- URLs are fetched and analyzed
- Metadata is tracked across all phases
- LLM costs are calculated
- No timeouts or truncated output

## Manual CLI Testing

### 1. Image Processing Test

```bash
export GOOGLE_API_KEY="your-api-key-here"

# Test with single image
msa "Analyze the financial data in this chart and suggest improvements" \
  --image test_multimodal_data/test_chart.png \
  --evolve \
  --traditional \
  --generations 2 \
  --population 2

# Test with multiple images
msa "Compare these two designs" \
  --image test_multimodal_data/chart1.png \
  --image test_multimodal_data/chart2.png \
  --evolve \
  --traditional \
  --generations 2 \
  --population 2

# Test with short form -i
msa "What insights can you draw from this image?" \
  -i test_multimodal_data/test_chart.png \
  --evolve \
  --traditional \
  --generations 2 \
  --population 2
```

**Verification Checklist:**
- [ ] Command runs without errors
- [ ] Output shows "📎 Processed: N images"
- [ ] QADI cycle completes all 4 phases
- [ ] Hypotheses reference image content
- [ ] Final answer incorporates image analysis
- [ ] No timeout errors
- [ ] LLM cost is displayed
- [ ] Evolution completes successfully

### 2. Document Processing Test

```bash
# Test with PDF document (requires a test PDF)
msa "Summarize the key findings in this research paper" \
  --document test_multimodal_data/research_paper.pdf \
  --evolve \
  --traditional \
  --generations 2 \
  --population 2

# Test with multiple documents
msa "Compare the methodologies in these two papers" \
  --document test_multimodal_data/paper1.pdf \
  --document test_multimodal_data/paper2.pdf \
  --evolve \
  --traditional \
  --generations 2 \
  --population 2

# Test with short form -d
msa "What are the main conclusions?" \
  -d test_multimodal_data/report.pdf \
  --evolve \
  --traditional \
  --generations 2 \
  --population 2
```

**Verification Checklist:**
- [ ] Command runs without errors
- [ ] Output shows "📎 Processed: N pages"
- [ ] Document content is analyzed
- [ ] Hypotheses reference document findings
- [ ] No timeout errors
- [ ] Evolution completes successfully

### 3. URL Processing Test

```bash
# Test with single URL
msa "What are the key points from this article?" \
  --url https://www.example.com/article \
  --evolve \
  --traditional \
  --generations 2 \
  --population 2

# Test with multiple URLs
msa "Compare the perspectives in these articles" \
  --url https://example1.com/article \
  --url https://example2.com/article \
  --evolve \
  --traditional \
  --generations 2 \
  --population 2

# Test with short form -u
msa "Summarize the main argument" \
  -u https://www.example.com/opinion \
  --evolve \
  --traditional \
  --generations 2 \
  --population 2
```

**Verification Checklist:**
- [ ] Command runs without errors
- [ ] Output shows "📎 Processed: N URLs"
- [ ] URL content is fetched and analyzed
- [ ] Hypotheses reference web content
- [ ] No timeout errors
- [ ] Evolution completes successfully

### 4. Combined Multimodal Test

```bash
# Test with all modalities together
msa "Synthesize insights from all these sources" \
  --image test_multimodal_data/chart.png \
  --document test_multimodal_data/report.pdf \
  --url https://www.example.com/article \
  --evolve \
  --traditional \
  --generations 2 \
  --population 2

# Test complex scenario with evolution
msa "How can we improve this product based on market data, user feedback, and competitor analysis?" \
  --image test_multimodal_data/market_chart.png \
  --image test_multimodal_data/user_feedback.png \
  --document test_multimodal_data/competitor_report.pdf \
  --url https://www.example.com/industry-trends \
  --url https://www.example.com/market-research \
  --evolve \
  --generations 3 \
  --population 5
```

**Verification Checklist:**
- [ ] Command runs without errors
- [ ] Output shows all modality types processed
- [ ] QADI integrates insights from all sources
- [ ] Hypotheses synthesize cross-modal information
- [ ] Evolution improves ideas across generations
- [ ] No timeout errors
- [ ] Final output is comprehensive and actionable

## Error Handling Tests

### 1. Invalid Inputs

```bash
# Test with nonexistent image
msa "Test" --image /nonexistent/image.png
# Expected: Error message about file not found

# Test with invalid URL
msa "Test" --url not-a-valid-url
# Expected: RuntimeError about invalid URL

# Test with too many URLs
msa "Test" \
  --url https://example1.com \
  --url https://example2.com \
  # ... (repeat 25 times)
# Expected: RuntimeError about too many URLs

# Test with unsupported image type
msa "Test" --image test.bmp
# Expected: RuntimeError about unsupported image type
```

**Verification Checklist:**
- [ ] Clear error messages for each failure case
- [ ] No stack traces or cryptic errors
- [ ] Suggestions for fixing the error
- [ ] Graceful degradation (no crashes)

### 2. Edge Cases

```bash
# Test with empty problem statement
msa "" --image test_chart.png
# Expected: Prompt for problem statement

# Test with very long problem statement (>1000 chars)
msa "$(python3 -c 'print("a" * 1500)')" --image test_chart.png
# Expected: Handles gracefully or provides clear error

# Test with special characters in filenames
msa "Test" --image "test file with spaces.png"
# Expected: Handles path correctly
```

## Performance Tests

### 1. Timeout Handling

```bash
# Test with multiple modalities and higher population
# Should complete within timeout or provide clear error
msa "Complex analysis task" \
  --image img1.png --image img2.png --image img3.png \
  --document doc1.pdf --document doc2.pdf \
  --url https://url1.com --url https://url2.com \
  --evolve \
  --generations 3 \
  --population 8
```

**Verification Checklist:**
- [ ] Completes within reasonable time (<5 minutes)
- [ ] Progress indicators show activity
- [ ] If timeout, clear error message
- [ ] Partial results are saved (if applicable)

### 2. Cost Tracking

```bash
# Run with cost tracking enabled
msa "Analysis task" \
  --image test_chart.png \
  --url https://example.com \
  --evolve \
  --traditional \
  --generations 2 \
  --population 3
```

**Verification Checklist:**
- [ ] Total LLM cost is displayed
- [ ] Cost is non-zero and reasonable
- [ ] Cost increases with more generations/population
- [ ] Multimodal processing cost is included

## Output Quality Checks

For each test above, verify:

1. **QADI Cycle Completeness**
   - [ ] Phase 1 (Questioning): Core question extracted
   - [ ] Phase 2 (Abduction): Multiple hypotheses generated
   - [ ] Phase 3 (Deduction): Hypotheses scored and ranked
   - [ ] Phase 4 (Induction): Final answer synthesized

2. **Multimodal Integration**
   - [ ] Image content referenced in hypotheses
   - [ ] Document findings incorporated
   - [ ] URL information integrated
   - [ ] Cross-modal synthesis in final answer

3. **Evolution Quality**
   - [ ] Ideas improve across generations
   - [ ] Diversity maintained in population
   - [ ] Best ideas clearly identified
   - [ ] Fitness scores meaningful

4. **User Experience**
   - [ ] Clear progress indicators
   - [ ] Helpful error messages
   - [ ] Output is well-formatted
   - [ ] Response times are reasonable

## Regression Tests

After any changes to multimodal code, run:

```bash
# Full test suite
uv run pytest tests/ -m "not integration" -v

# Multimodal-specific tests
uv run pytest tests/test_cli_multimodal.py tests/core/test_orchestrator_multimodal_signatures.py -v

# With real API (if available)
GOOGLE_API_KEY=xxx uv run pytest tests/test_real_api_multimodal.py -v
```

## Success Criteria

The multimodal support is considered fully working when:

- [ ] All unit tests pass (100%)
- [ ] All integration tests pass (with API key)
- [ ] All manual CLI tests complete successfully
- [ ] Error handling works as expected
- [ ] Performance is acceptable (<5min for typical usage)
- [ ] Documentation is clear and complete
- [ ] No regressions in existing functionality

## Troubleshooting

### Common Issues

1. **"GOOGLE_API_KEY not set"**
   - Solution: Export the API key in your shell: `export GOOGLE_API_KEY="xxx"`

2. **"FileNotFoundError" for images**
   - Solution: Use absolute paths or verify file exists
   - Check: `ls -la test_multimodal_data/test_chart.png`

3. **"Timeout" errors**
   - Solution: Reduce population size or generations
   - Try: `--generations 2 --population 2`

4. **"RuntimeError: Invalid URL"**
   - Solution: Ensure URL starts with http:// or https://
   - Use: `--url https://www.example.com`

5. **Integration tests skip with "GOOGLE_API_KEY not available"**
   - This is expected behavior when API key is not set
   - Tests will run when key is provided

## Next Steps

After completing all tests:

1. Document any issues found
2. Update CLAUDE.md with any learned patterns
3. Update README with final usage examples
4. Create PR for review
5. Run CI/CD pipeline to ensure no regressions
