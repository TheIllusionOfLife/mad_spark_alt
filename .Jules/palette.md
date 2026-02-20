## 2025-05-23 - Conditional Rich Progress Bars
**Learning:** Using `contextlib.nullcontext()` conditionally with `rich.progress.Progress` allows for clean handling of verbose/silent modes without nested `if/else` blocks.
**Action:** When adding progress bars to CLI commands, define `progress_ctx = nullcontext() if verbose else Progress(...)` and use `with progress_ctx as progress:` to wrap the operation. Check `if progress:` before adding tasks.
