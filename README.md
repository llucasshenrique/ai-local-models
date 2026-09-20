# Local LLM workers

Tooling to run small, well-specified code micro-tasks on local models (ollama + opencode) on an
11 GB GPU. Findings and recommendations go in `docs/local-llm-workers.md`; the task spec is `SPEC.md`.

| Path | Contents |
|---|---|
| `modelfiles/` | Tuned ollama Modelfiles (`<name>.<tag>.Modelfile` becomes `<name>:<tag>`) |
| `scripts/` | `create-models.sh` (idempotent `ollama create`), `run-benchmarks.sh`, `run-extra.sh` (re-run given model/tasks) |
| `bench/` | `bench.py` (tok/s + GPU fit), `smoke.py` (3-task loop smoke test), `sweep.py` (model x quantization sweep) |
| `harness/` | `minimal_agent.py`: ~100-line tool-loop baseline harness |
| `results/` | Raw JSONL results (`opencode.jsonl`, `sweep.jsonl`, `sweep-tasks.jsonl`) |
| `docs/` | Write-up |

Run from the repo root, e.g. `scripts/create-models.sh`, `python3 bench/smoke.py opencode gemma4-agent:12b 1 2 3`.
