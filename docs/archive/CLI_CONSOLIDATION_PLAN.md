# CLI Consolidation Implementation Plan

**Status**: In Progress
**Branch**: `feature/unified-cli-with-default-command`
**Created**: 2025-11-09

## Goal
Consolidate all features into a single `mad_spark_alt` Click-based CLI with Rich styling, where QADI is the default command (no subcommand needed), then delete `mad-spark` and `cli.py`.

## CLI Structure

```bash
# Default: QADI analysis (no subcommand needed)
uv run mad_spark_alt "Your question"
msa "Your question"  # Short alias

# QADI with options
uv run mad_spark_alt "Question" --evolve --image file.png
msa "Question" --temperature 1.2 --verbose

# Explicit subcommands for other features
uv run mad_spark_alt evolve "Problem" --generations 3
uv run mad_spark_alt evaluate "Text"
uv run mad_spark_alt batch-evaluate file1.txt file2.txt
uv run mad_spark_alt compare "Prompt" -r "Response 1" -r "Response 2"
uv run mad_spark_alt list-evaluators
```

## Implementation Checklist (31 Tasks)

### Phase 1: TDD - Write Tests First
- [x] 1. Create feature branch from up-to-date main
- [x] 2. Write tests for unified CLI default QADI command
- [x] 3. Write tests for multimodal support in default command
- [x] 4. Write tests for evolve subcommand
- [x] 5. Write tests for evaluation subcommands

### Phase 2: Implementation
- [ ] 6. Create unified_cli.py with default QADI command
- [ ] 7. Add multimodal processing to default command
- [ ] 8. Implement evolve subcommand
- [ ] 9. Implement evaluation subcommands
- [ ] 10. Run tests and fix failures
- [ ] 11. Update pyproject.toml entry points (add msa alias)

### Phase 3: Test Migration
- [ ] 12. Migrate all 17 qadi_simple tests to Click CLI
- [ ] 13. Migrate 5 cli.py tests to unified CLI

### Phase 4: Real API Testing (Critical - No Mock Mode)
- [ ] 14. Test with real API: basic QADI analysis
- [ ] 15. Test with real API: QADI with evolution
- [ ] 16. Test with real API: multimodal (image + document + URL)
- [ ] 17. Test with real API: all evaluation subcommands
- [ ] 18. Verify short alias 'msa' works

### Phase 5: Documentation Updates
- [ ] 19. Update README.md with new CLI structure
- [ ] 20. Update docs/cli_usage.md
- [ ] 21. Update docs/MULTIMODAL_TESTING_GUIDE.md
- [ ] 22. Update ARCHITECTURE.md CLI section
- [ ] 23. Create docs/CLI_MIGRATION.md guide
- [ ] 24. Update all help text and error messages

### Phase 6: Cleanup & Verification
- [ ] 25. Delete old files (cli.py, qadi_simple.py, main_qadi.py)
- [ ] 26. Run full test suite (all 918+ tests)
- [ ] 27. Manual user testing: all command variations
- [ ] 28. Verify no timeouts, truncation, errors, or format issues
- [ ] 29. Update CI tests if needed

### Phase 7: Git & PR
- [ ] 30. Create commits following conventional commit format
- [ ] 31. Push to GitHub and create PR

## Quality Criteria (Must All Pass)

✅ **No Timeouts**: All commands complete within reasonable time
✅ **No Truncation**: Full output displayed
✅ **No Errors**: All commands work correctly
✅ **No Repeated Content**: Output is clean and unique
✅ **Proper Format**: Rich styling works, tables display correctly
✅ **All Tests Pass**: 918+ tests passing
✅ **Real API Works**: Tested with actual GOOGLE_API_KEY
✅ **Docs Updated**: All documentation reflects new structure

## File Changes Summary

### New Files
- `src/mad_spark_alt/unified_cli.py` (~1500 lines)
- `tests/test_unified_cli.py` (~300 lines)
- `docs/CLI_MIGRATION.md` (~200 lines)

### Modified Files
- `pyproject.toml` (add msa alias, remove mad-spark)
- `README.md` (update CLI examples)
- `docs/cli_usage.md` (complete rewrite)
- `docs/MULTIMODAL_TESTING_GUIDE.md` (update command references)
- `ARCHITECTURE.md` (update CLI section)
- ~17 test files (qadi_simple tests migration)
- ~5 test files (cli.py tests migration)

### Deleted Files
- `src/mad_spark_alt/cli.py` (984 lines)
- `qadi_simple.py` (1036 lines)
- `src/mad_spark_alt/main_qadi.py` (20 lines)

## Key Implementation Details

### Click Structure
```python
@click.group(invoke_without_command=True)
@click.pass_context
@click.argument('input', required=False)
@click.option('--verbose', '-v', is_flag=True)
@click.option('--temperature', '-t', type=float)
@click.option('--evolve', '-e', is_flag=True)
@click.option('--generations', '-g', type=int, default=2)
@click.option('--population', '-p', type=int, default=5)
@click.option('--image', '-i', multiple=True, type=click.Path(exists=True))
@click.option('--document', '-d', multiple=True, type=click.Path(exists=True))
@click.option('--url', '-u', multiple=True)
def main(ctx, input, ...):
    if ctx.invoked_subcommand is None:
        # Run default QADI analysis
        asyncio.run(_run_qadi_analysis(...))
```

### Test Migration Pattern
```python
# OLD (argparse)
with patch('sys.argv', ['qadi_simple.py', 'test', '--evolve']):
    main()

# NEW (Click)
from click.testing import CliRunner
runner = CliRunner()
result = runner.invoke(main, ['test', '--evolve'])
```

## Success Metrics

- **Single Entry Point**: Only `mad_spark_alt` and `msa`
- **Default Command**: Works without subcommand
- **All Features**: 6 commands total (default + 5 subcommands)
- **Zero Regressions**: All existing functionality preserved
- **Quality Output**: No timeouts, truncation, or errors
- **Complete Docs**: All documentation updated

## Timeline

- Phase 1-2: Implementation (6 hours)
- Phase 3: Test Migration (3 hours)
- Phase 4: Real API Testing (2 hours)
- Phase 5: Documentation (3 hours)
- Phase 6-7: Cleanup & PR (2 hours)

**Total**: ~16 hours of focused work
