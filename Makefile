# llmeval shortcuts. `make help` lists them. Config: CONFIG=evals/default.toml (override with CONFIG=...)
CONFIG ?= evals/default.toml
PY     ?= python3
EVAL   := $(PY) -m llmeval -c $(CONFIG)
DEMO   := /tmp/llmeval-demo

.PHONY: help models add test tui tui-demo list selfcheck prepare run fit report tune clean-demo

help:  ## show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | column -t -s "$$(printf '\t')"

models:  ## installed ollama models and which are in the config
	$(EVAL) models

add:  ## add installed models to the config: make add MODEL="granite4.1:8b qwen3:8b"  (PULL=1 downloads, BASE=x builds a tuned tag)
	@test -n "$(MODEL)" || { echo 'usage: make add MODEL="tag1 tag2" [PULL=1] [BASE=base:tag CTX=32768]'; exit 2; }
	$(EVAL) add $(MODEL) $(if $(PULL),--pull) $(if $(BASE),--base $(BASE)) $(if $(CTX),--ctx $(CTX))

test:  ## unit tests + TUI pty test (no GPU, throw-away data)
	$(PY) -m unittest discover -s tests -v

tui-demo:  ## try the TUI on synthetic data: no GPU, no ollama, your results untouched
	rm -rf $(DEMO) && mkdir -p $(DEMO) && LLMEVAL_RESULTS=$(DEMO) $(PY) -m llmeval demo-data
	LLMEVAL_RESULTS=$(DEMO) $(EVAL) tui

tui:  ## TUI on your real results (tab 2: r = start run, tab 3: m = measure fit)
	$(EVAL) tui

list:  ## harnesses installed, tasks, ollama status
	$(EVAL) list

selfcheck:  ## prove every task is solvable (no GPU)
	$(EVAL) selfcheck

prepare:  ## pull base models and create tuned tags from the config
	$(EVAL) prepare

run:  ## run the model x harness x task matrix (resumable)
	$(EVAL) run

fit:  ## GPU fit and tok/s at 16k..64k context per model
	$(EVAL) fit

report:  ## markdown ranking of all results
	$(EVAL) report

tune:  ## self-improvement search: make tune MODEL=ornith-agent:9b
	@test -n "$(MODEL)" || { echo "usage: make tune MODEL=<tag with a base in the config>"; exit 2; }
	$(EVAL) tune $(MODEL)

clean-demo:  ## remove the demo data
	rm -rf $(DEMO)
