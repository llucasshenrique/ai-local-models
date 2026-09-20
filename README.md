# llmeval: evaluate local models and coding-agent harnesses

Automatically measures which **model x harness** pairs can complete small, test-verified code tasks on your machine,
and how much time, tool calls and prompt context they cost. Stdlib only (Python 3.11+), works with ollama.
Background and findings: `docs/local-llm-workers.md`; original task spec: `SPEC.md`.

```
bin/llmeval list          # harnesses installed? tasks? ollama reachable?
bin/llmeval selfcheck     # prove every task is solvable and starts failing (no GPU)
bin/llmeval prepare       # pull base models, create tuned tags from evals/default.toml
bin/llmeval run           # run the matrix; resumable, results append to results/runs.jsonl
bin/llmeval fit           # GPU fit + tok/s per model at 16k..64k context
bin/llmeval report        # markdown ranking (--out file.md)
bin/llmeval tui           # terminal UI: results, live runs, context fit, setup, tune
bin/llmeval tune MODEL    # bounded self-improvement search over Modelfile params (see below)
bin/llmeval import-legacy # import the pre-llmeval results (marked legacy)
```

## How it works

- **Tasks are data** (`tasks/*.json`: prompt, starting files, test command, protected files, reference solution).
  A trial passes only if the test passes **and** the protected files (the tests) are untouched.
- **Harness adapters** (`llmeval/harnesses.py`): opencode, minimal (`harness/minimal_agent.py`), mini-swe-agent, aider, Pi,
  plus experimental Qwen Code / Codex / Cline via `ollama launch`. Adding one = one small class.
- **Uniform metrics:** requests and peak prompt tokens are read by an in-process proxy (`llmeval/proxy.py`) from ollama's
  own counters, so every harness is measured the same way (opencode reports the same numbers in its event stream).
  Loops = the same tool call with identical arguments 3 times (the run is killed); reported where the harness exposes tool calls.
- **One GPU lock** (`/tmp/local-llm-gpu.lock`) serializes every job that loads a model; the UI never loads one itself.
- **Reproducible rows:** each trial records the ollama version and the model digest. Bad data is moved to
  `results/invalid/` with a reason (`store.invalidate`), never silently edited. Per-trial output goes to `results/traces/`.
- **Safe by default:** only tags named in the config are created; nothing is deleted except `tune-*` candidates the tuner
  itself created. Harness configs (opencode, Pi) are backed up once as `*.bak-llmeval` before the first edit.

## Config (`evals/default.toml`)

`[run]` harnesses, reps, timeout, tasks; `[defaults.params]` Modelfile parameters; `[[models]]` with `tag` (+ optional `base`
to build it from, + `[models.params]` overrides). A model without `base` is used as-is.

## Self-improvement (`tune`)

The eval is the fitness function for a coordinate search over `temperature, top_k, top_p, repeat_penalty, num_ctx`.
Guardrails: candidates score on **train** tasks; a winner must also beat the baseline on **held-out** tasks
(`[tune] holdout = [...]`, default `05-bug-across-files`) before anything is proposed; the output is a proposal file
`results/tune-best.toml`, never an automatic config change. Changing the *harness* prompt or letting a model edit its own
tooling is deliberately out of scope until the parameter loop proves trustworthy.

## Layout

`llmeval/` package | `tasks/` task definitions | `evals/` configs | `harness/minimal_agent.py` | `modelfiles/` hand-written and
historical Modelfiles | `config/` snapshots of the opencode/Pi configs | `results/` data | `tests/` unit tests
(`python3 -m unittest discover -s tests`) | `docs/` write-up.
