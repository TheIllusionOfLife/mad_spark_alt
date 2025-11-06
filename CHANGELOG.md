# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed
- **Breaking Change**: Removed deprecated modules `prompt_classifier` and `adaptive_prompts` (1,253 lines)
  - `src/mad_spark_alt/core/prompt_classifier.py` (749 lines)
  - `src/mad_spark_alt/core/adaptive_prompts.py` (504 lines)
  - These modules were deprecated in favor of `SimpleQADIOrchestrator`
  - Migration guide available in `DEPRECATED.md`
- Removed deprecated script `qadi_simple_multi.py` (568 lines)
  - Use CLI command `uv run mad-spark qadi` instead
- **Total Impact**: -1,821 lines (-11% codebase size)

### Documentation
- Updated all documentation to remove references to deprecated modules
- Added comprehensive migration guide in `DEPRECATED.md`
- Updated CLI usage examples to use modern commands

### Migration
See `DEPRECATED.md` for detailed migration instructions from:
- `prompt_classifier`/`adaptive_prompts` → `SimpleQADIOrchestrator`
- `qadi_simple_multi.py` → `uv run mad-spark qadi`

---

## [Previous Releases]

For release history before this changelog, see git commit history.
