## 2024-05-22 - Sequential Loop Overhead in Genetic Algorithm
**Learning:** Found a classic performance anti-pattern where an async batch-capable function (`evaluate_population`) was being called inside a sequential loop, effectively disabling its batching capabilities and serializing operations. This caused N sequential API calls instead of 1 parallel batch call.
**Action:** Always check if operations inside a loop can be collected and executed in a single batch call, especially for IO-bound tasks like LLM evaluations. Prefer `collect -> batch_process` over `loop { process }`.

## 2024-05-22 - Python Random Selection Optimization
**Learning:** `random.choices` (Python 3.6+) is significantly faster (O(n) vs O(k*n)) than manual cumulative weight loops for weighted selection, especially with large populations.
**Action:** Always prefer `random.choices` for weighted sampling instead of implementing custom roulette wheel logic.
