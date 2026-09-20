# RSI Protocol: Reliable Recursive Self-Improvement for Local LLM Coding Agents

> **Core Principle:** The system may optimize the subject being evaluated, but must not silently optimize the measurement system itself.

## 1. Immutable Benchmark Infrastructure (Frozen)
The ruler must remain immutable during model and configuration self-improvement. Never edit or weaken:
- **Task definitions:** `tasks/*.json`
- **Evaluation & scoring:** `llmeval/tasks.py`, `llmeval/runner.py`, `llmeval/sandbox.py`, `llmeval/guardrails.py`
- **Result schema & persistence:** `llmeval/store.py`, `llmeval/provenance.py`
- **Statistical analysis & Pareto:** `llmeval/stats.py`, `llmeval/pareto.py`
- **Reporting & comparison logic:** `llmeval/report.py`
- **Benchmark configuration & tests:** `evals/default.toml`, `tests/` (new tests may be added, existing must not be weakened), `SPEC.md`, `RSI.md`.

Cryptographic integrity checks (`llmeval/guardrails.py`) verify benchmark immutability at every run.

## 2. Subject Under Optimization (Allowed)
- **Model Parameters:** Temperature, top_k, top_p, repeat_penalty, num_predict, seed via Modelfile/Ollama.
- **Context Size:** Benchmarked explicitly across hardware-safe context envelopes (`--ctx-sweep`).
- **Quantizations & Model Sizes:** Explored via families and Pareto selection.
- **Harness Optimizations:** Dedicated harness-optimization experiments may modify `harness/minimal_agent.py` under an isolated, dedicated experiment mode with its own baseline.

## 3. Staged Evaluation & Statistical Promotion Rules
A configuration cannot become the new baseline merely because it won a noisy low-repetition trial.
1. **Stage A — Screening / Exploration:** Low repetition exploration on `train` tasks.
2. **Stage B — Confirmation:** Candidates with preliminary gains must be verified with higher repetitions (`--confirm-reps`, min 5-10 reps) on `train` tasks.
3. **Stage C — Held-out Validation:** Confirmed candidates must show non-regression on independent `val` (holdout) tasks.
4. **Stage D — Final Confirmation:** Full benchmark suite evaluation. Final `test` tasks are strictly isolated and never participate in tuner search decisions.

**Promotion Rule:** Promotion requires statistical evidence (`improvement_supported` decision based on Wilson confidence intervals and statistical tests). If evidence is insufficient, the system reports `insufficient_evidence` and rejects promotion.

## 4. Hardware Fit vs Quality Optimization Separation
- Hardware context fitting (`fit`, `fit-opt`) determines **physical runtime feasibility** (`hardware_safe_ctx`, `hardware_max_ctx`, VRAM boundary, swap thrashing, CPU load).
- It does **NOT** claim that physical maximum is quality-optimal.
- Quality-optimal context must be experimentally measured on the benchmark (`llmeval run --ctx-sweep`).

## 5. Workspace Isolation & Sandboxing
- Each trial runs in a disposable temporary workspace.
- Protected evaluation files (`test.sh`) are enforced read-only (`0444`).
- Child processes are bounded by process groups, timeouts, and resource limits (`RLIMIT_CPU`, `RLIMIT_FSIZE`, `RLIMIT_NPROC`).
