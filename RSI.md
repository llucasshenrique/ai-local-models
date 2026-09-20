# RSI protocol (recursive self-improvement of llmeval)

You are a local model (ornith-agent:9b) improving the project you are running in. Be careful: you must not change the ruler
that measures you. Work in small, verified steps.

## Frozen (never edit): the evaluation
`llmeval/tasks.py`, `llmeval/runner.py` (only the two additive `--variant` changes already there), `llmeval/store.py`,
`llmeval/report.py`, `llmeval/prune.py`, everything in `tests/` that exists now (you may ADD new test files), `tasks/*.json`,
`evals/default.toml`, `results/`, `SPEC.md`, this file. Never weaken a test or a task.

## Allowed
`harness/minimal_agent.py` (its system prompt and loop), new files, `docs/`, `README.md`, new evals under `evals/`.

## Hard rules
- Stay on this branch. Do not touch `~/.config`, `~/.pi`, ollama models (no `ollama pull/rm/create`), or any file outside this repo.
- Use ONLY the model `ornith-agent:9b` (it is already loaded for you; other models would evict it).
- Never edit or delete `results/`. Do not run more than one `llmeval run` at a time.
- After every change run `make test`; if it fails, revert the change.

## The loop (max 3 iterations, then stop and report)
1. Baseline once: `python3 -m llmeval -c evals/rsi.toml run --variant base` (create evals/rsi.toml first if missing: harness minimal,
   model ornith-agent:9b (no base), reps 3, timeout 90).
2. Pick ONE small change to `harness/minimal_agent.py` (e.g. a clearer system prompt, better error text back to the model).
3. `python3 -m llmeval -c evals/rsi.toml run --variant v1` then `python3 -m llmeval compare base v1 --harness minimal`.
4. Verdict `better`: commit with the numbers in the message. `same` or `worse`: `git checkout harness/minimal_agent.py`.
5. Append what you tried and the verdict to `RSI_LOG.md`. Next iteration uses variant names v2, v3.

## Final report
Write `RSI_LOG.md`: what changed, before/after numbers, what you would try next, anything you were unsure about. Do not merge.
