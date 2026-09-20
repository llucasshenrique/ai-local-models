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
bin/llmeval fit --optimize # empirically discover max safe context (RSI loop, no LLM required)
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

Recommendation rule (`llmeval/families.py` & `llmeval/pareto.py`): computes the true **Pareto frontier** across quality (pass rate), latency (wall time), model size, and failure count. By default under the `"balanced"` policy, it recommends the smallest variant within tolerance of the top pass rate among GPU-fitting candidates, tie-breaking by speed. Alternative policies (`--policy max_quality`, `--policy fastest`, `--policy pareto`) are also supported.

## Context evaluation: physical feasibility vs. quality optimization

llmeval explicitly separates **physical runtime feasibility** from **quality-optimal context**:
1. `llmeval fit --optimize`: determines the physical hardware boundary (VRAM limits, zero swap thrashing, 100% GPU offload). This establishes the safe context envelope.
2. `llmeval run --ctx-sweep "16384,32768,65536"`: experimentally measures model coding pass rate, latency, and tool calls across context sizes to detect regressions and determine the quality-optimal context size.

Instead of guessing context limits from model metadata or requiring an LLM to orchestrate measurements, `llmeval fit --optimize` executes an autonomous empirical loop:

```
make fit-opt MODEL=ornith-agent:9b           # run adaptive search and output recommendations
make fit-opt MODEL=ornith-agent:9b APPLY=1   # optimize and re-create the tag with the safe num_ctx
```

What it does autonomously:
1. **Background Telemetry**: Samples GPU VRAM, GPU layer offloading, host RAM, swap deltas, and CPU load during inference.
2. **Realistic Context Stress**: Fills 75%–85% of each tested context window to measure real prefill throughput and KV allocation, not a 20-token toy prompt.
3. **Adaptive Search Progression**: Starts at baseline, doubles context coarsely, detects GPU/swap boundary violations, bisects the boundary via binary search, and runs high-stress validation.
4. **Safety Margin**: Recommends a production context size with headroom for desktop GPU display buffers and multi-turn KV caches. Results append to `results/fit.jsonl`.


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
