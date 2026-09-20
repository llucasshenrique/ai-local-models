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

## Trying it

```
make help        # all shortcuts
make tui-demo    # TUI on synthetic data: no GPU, no ollama, your real results untouched
make test        # unit tests + a scripted TUI session in a pseudo-terminal
make tui         # TUI on your real results (tab 2: r = run, tab 3: m = fit, tab 4: s = selfcheck / p = prepare)
```

TUI keys: `1-5` tabs, arrows/PgUp/PgDn scroll, `x` stop after the current trial, `q` quit.
`results/` is local data and git-ignored; set `LLMEVAL_RESULTS=/some/dir` to use another location.

## Adding models

```
make models                                   # what ollama has installed, and which are already in the config
make add MODEL="qwen3:8b granite4.1:3b"       # use installed models as-is (added to evals/default.toml)
make add MODEL="granite4.1:8b" PULL=1         # download first, then add
make add MODEL=my-agent:9b BASE=ornith:9b-q4_K_M CTX=32768   # build a tuned tag from a base (created by `make prepare`)
```

Or edit `evals/default.toml` by hand: a `[[models]]` entry with only `tag` uses an existing model unchanged; adding `base`
(and optional `[models.params]`) makes `prepare` create it with the tuned Modelfile parameters.
Then `make run` only runs what is missing (finished trials are skipped).

## Model families: choose between sizes and quantizations

Describe a family once and llmeval compares every size x quantization variant, then recommends one:

```
make add-family NAME=granite4.1 SIZES=3b,8b QUANTS=q4_K_M,q6_K,q8_0 CTX=32768   # or key f in the TUI (Models tab)
python3 -m llmeval prepare --family granite4.1 --dry-run    # what would be downloaded (nothing is pulled)
python3 -m llmeval prepare --family granite4.1              # pull + create tuned <family>-agent:<size>-<quant> tags
python3 -m llmeval run --family granite4.1                  # only that family
make fit && make families                                   # add VRAM/tok/s, then see the recommendation
```

Recommendation rule (`llmeval/families.py`): among variants that fit fully on the GPU, take the best pass rate, then pick the
**smallest** variant (parameters, then VRAM) within 5 points of it, ties by speed. The report and the TUI Results tab show it.

## Advisor: use the best model to think about improvements

`make advise` (TUI: Tune tab, key `a`) sends a compact digest of your results, family tables, tune history and a few failure
traces to the best model in your results (or `[advisor] model` in the config) and saves `results/advice.md`.
It only proposes **hypotheses**: answers are validated (known models, allowed parameters, safe ranges) and never applied.
Test them with `python3 -m llmeval tune MODEL --from-advice`, where measurement (train + held-out tasks) decides.
`make advise ARGS=--dry-run` shows the prompt without calling any model. Small samples fool LLMs too (the advisor once called a
1-trial vs 3-trial difference "consistent"), which is exactly why nothing it says is trusted without a measurement.

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
historical Modelfiles | `config/` snapshots of the opencode/Pi configs | `results/` local data (git-ignored) | `tests/` unit tests
(`python3 -m unittest discover -s tests`) | `docs/` write-up.
