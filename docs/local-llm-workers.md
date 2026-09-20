# Local LLM workers on an 11 GB GPU (ollama + Orca)

Goal: run small, single-file, well-specified code micro-tasks (with a test command) on local models, as a
native Orca agent, without loops. Hardware: RTX 2080 Ti 11 GB, 31 GB RAM, ollama 0.31.1. Raw data is in `results/`,
tooling in `bench/`, `scripts/`, `harness/`, `modelfiles/`; configs in `config/`.

## TL;DR

- **Recommended pair: Pi (native Orca agent) + `ornith-agent:9b`** (ornith 9B q4_K_M, 32k ctx, ~66 tok/s, 6.3 GB VRAM, 100% GPU).
  It passed 9/9 opencode runs and 3/3 Pi runs, and is the fastest model that still holds 64k context on the GPU.
- Runner-up model: `gemma4-agent:12b` (9/9, slightly steadier timing, ~45 tok/s, 8.1 GB, 64k fits).
- **The gemma4 "loops/inefficiency" was mostly context, not sampling:** ollama's default context here is 4096, while
  opencode alone sends ~11k tokens of prompt. Raising `num_ctx` fixed it; tuned sampling alone did not (ablation).
- opencode costs ~11k tokens of context before the task starts. Lighter harnesses (Pi ~4-5k, mini-swe-agent ~2k,
  aider ~0.9k, our minimal loop ~0.5-0.8k) all passed the same tasks.

## Results

### Model / quantization sweep (opencode smoke test, 3 tasks, 1 run each)
| Base tag (quant) | Pass /3 | tok/s | VRAM / GPU fit | Wall 3 tasks (s) | Loop flags |
|---|---|---|---|---|---|
| ornith:9b-q4_K_M | 3 | 66.8 | 5.7 GB 100% GPU | 225.0 | 0 |
| ornith:9b-q8_0 | 3 | 43.9 | 9.1 GB 100% GPU | 264.6 | 0 |
| granite4.1:8b-q3_K_M | 1 | 70.0 | 7.1 GB 100% GPU | 128.0 | 0 |
| granite4.1:8b-q4_K_M | 3 | 71.4 | 7.9 GB 100% GPU | 141.4 | 1 |
| granite4.1:8b-q6_K | 3 | 35.6 | 10 GB 8%/92% CPU/GPU | 182.9 | 0 |
| granite4.1:3b-q4_K_M | 2 | 126.8 | 3.6 GB 100% GPU | 247.6 | 0 |
| granite4.1:3b-q8_0 | 3 | 60.4 | 5.1 GB 100% GPU | 113.9 | 0 |
| qwen3:8b-q8_0 | 0 | 14.4 | 11 GB 21%/79% CPU/GPU | 470.2 | 0 |

qwen3:14b (9.3 GB) never fits: 76-84% GPU, 11-17 tok/s, 0/3 within the 180 s cap. Quantization findings: q3_K_M
lost accuracy (granite 8B 1/3); q6_K/q8_0 of 8-9B models overflow 11 GB and slow down; q4_K_M is the sweet spot.

### Finalists repeated 3x (opencode, 3 tasks x 3 reps = 9 runs each)
| Model | Pass | Median wall (s) | Max wall (s) | Loop flags | Median tool calls |
|---|---|---|---|---|---|
| gemma4-agent:12b | 9/9 | 42 | 44 | 0 | 4 |
| sw-ornith-9b-q4-k-m | 9/9 | 54 | 85 | 0 | 5 |
| sw-granite4-1-8b-q4-k-m | 9/9 | 43 | 61 | 2 | 2 |
| qwen3-agent:8b | 3/9 | 61 | 94 | 1 | 3 |

"Loop flag" = the same tool call with identical arguments issued 3 times (the runner kills the run); for granite the
tests still passed, so those are likely benign retries of the test command.

### Context headroom (`num_ctx` vs GPU fit, one short generation each)
| Model | 16k | 32k | 48k | 64k |
|---|---|---|---|---|
| gemma4-agent:12b | 8.1 GB 100% GPU, 44.4 tok/s | 8.1 GB 100% GPU, 45.1 tok/s | 8.1 GB 100% GPU, 46.4 tok/s | 8.1 GB 100% GPU, 47.6 tok/s |
| sw-ornith-9b-q4-k-m | 5.7 GB 100% GPU, 64.8 tok/s | 6.3 GB 100% GPU, 66.1 tok/s | 6.9 GB 100% GPU, 66.4 tok/s | 7.4 GB 100% GPU, 67.3 tok/s |
| sw-granite4-1-8b-q4-k-m | 7.9 GB 100% GPU, 75.2 tok/s | 11 GB 14%/86% CPU/GPU, 29.5 tok/s | 13 GB 32%/68% CPU/GPU, 15.8 tok/s | 16 GB 44%/56% CPU/GPU, 12.4 tok/s |
| qwen3-agent:8b | 7.5 GB 100% GPU, 75.1 tok/s | 10 GB 9%/91% CPU/GPU, 44.4 tok/s | 11 GB 19%/81% CPU/GPU, 23.3 tok/s | 11 GB 19%/81% CPU/GPU, 24.8 tok/s |

gemma4 (sliding-window KV) and ornith keep 64k fully on the GPU. granite 8B and qwen3 8B only fit 16k.

### Flash attention / KV cache type (same host, temporary server, gemma4:12b)
| Variant | ctx | tok/s (2 runs) | Fit |
|---|---|---|---|
| default | 16384 | 45.5 / 45.1 | 8.6 GB 100% GPU |
| default | 32768 | 21.4 / 21.4 | 9.3 GB 11%/89% CPU/GPU |
| default | 49152 | 13.1 / 12.6 | 9.9 GB 20%/80% CPU/GPU |
| fa-q8 | 16384 | 43.7 / 46.1 | 7.8 GB 100% GPU |
| fa-q8 | 32768 | 44.6 / 45.9 | 7.9 GB 100% GPU |
| fa-q8 | 49152 | 44.8 / 44.2 | 7.9 GB 100% GPU |
| fa-only | 16384 | 48.0 / 47.1 | 8.1 GB 100% GPU |
| fa-only | 32768 | 43.8 / 45.2 | 8.1 GB 100% GPU |
| fa-only | 49152 | 47.3 / 47.8 | 8.1 GB 100% GPU |

Forcing flash attention **off** makes 32k+ spill to CPU (21 and 13 tok/s). With FA on, KV `q8_0` saves only ~0.2-0.3 GB
and does not change speed, so it is not worth configuring. The local systemd server already behaves like "FA on"
(auto), which is why 48k measured 100% GPU there.

### Ablation: context vs sampling (gemma4:12b, opencode)
| Variant | Pass /3 |
|---|---|
| gemma4:12b untuned (server default ctx = 4096) | 0/3 (first matrix; see Caveats) |
| abl-gemma4-ctxonly: untuned sampling, num_ctx 32768 | 3/3 |
| abl-gemma4-samponly: tuned sampling, default ctx 4096 | 0/3 |

Untuned `gemma4:12b` loads with CONTEXT 4096 (`results/ablation-ps.txt`), so opencode's ~11k-token prompt was truncated.

### Harness comparison (same 3 tasks; all but opencode measured through `bench/proxy.py`)
| Harness | Model | Pass | Median wall (s) | Median LLM requests | Peak prompt tokens (max) |
|---|---|---|---|---|---|
| aider | gemma4-agent:12b | 3/3 | 23.7 | 1 | 925 |
| aider | sw-ornith-9b-q4-k-m | 3/3 | 15.3 | 1 | 922 |
| mini | gemma4-agent:12b | 3/3 | 33.5 | 10 | 2352 |
| mini | sw-ornith-9b-q4-k-m | 3/3 | 18.2 | 7 | 2266 |
| minimal | gemma4-agent:12b | 3/3 | 7.8 | 6 | 492 |
| minimal | sw-ornith-9b-q4-k-m | 3/3 | 5.7 | 4 | 839 |
| pi | gemma4-agent:12b | 3/3 | 27.0 | 6 | 4323 |
| pi | sw-ornith-9b-q4-k-m | 3/3 | 15.9 | 5 | 4899 |

opencode reference (from the repeats above): 9/9 runs, ~11-12k peak prompt tokens, 42-54 s median. Requests/peak
tokens for opencode come from its event stream, not the proxy.

### First opencode matrix (untuned vs tuned, n=1 per task)
| Model (opencode, first matrix, n=1/task) | Pass /3 | Wall per task (s) |
|---|---|---|
| gemma4-agent:12b | 3/3 | 46.2, 151.9, 60.2 |
| gemma4:12b | 0/3 | 49.0, 97.9, 97.6 |
| qwen3-agent:14b | 0/3 | 180.2, 180.1, 180.2 |
| qwen3-agent:8b | 2/3 | 140.1, 59.9, 62.1 |


## Tuned tags (`modelfiles/`, created by `scripts/create-models.sh`)

| Tag | Base | Sampling | ctx / predict | Evidence |
|---|---|---|---|---|
| `ornith-agent:9b` | ornith:9b-q4_K_M | temp 0.25, top_k 20, top_p 0.9, repeat_penalty 1.05, repeat_last_n 256 | 32768 / 4096 | 6.3 GB, 100% GPU at 32k (7.4 GB at 64k), 9/9 |
| `gemma4-agent:12b` | gemma4:12b | temp 0.2, top_k 30, top_p 0.9, repeat_penalty 1.1, repeat_last_n 256 | 32768 / 4096 | 8.1 GB flat 16k-64k, 9/9 |
| `qwen3-agent:8b` / `:14b` | qwen3 | temp 0.25, top_k 20, top_p 0.9 | 16384 / 4096 | 3/9 and 0/3: not recommended |

Sampling values are conservative defaults for tool use (low temperature, mild repetition penalty); the ablation shows
context length was the decisive factor here, and no experiment isolates the benefit of the sampling values themselves.
Stop tokens are inherited from each base model's template (only qwen3 sets explicit `<|im_start|>/<|im_end|>`).
Thinking: ornith/gemma4 benches ran with thinking off; gemma4 with thinking on produced ~4x more tokens (714 vs 161)
for the same answer, so keep it off for micro-tasks (`defaultThinkingLevel: off` in Pi).

## Server environment (do not apply blindly; needs sudo)

The local systemd `ollama` service ran with defaults (`OLLAMA_CONTEXT_LENGTH` unset -> 4096). Recommended override:

```
sudo systemctl edit ollama      # then add:
[Service]
Environment="OLLAMA_CONTEXT_LENGTH=32768"
Environment="OLLAMA_KEEP_ALIVE=30m"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
sudo systemctl restart ollama
```

Do **not** set `OLLAMA_FLASH_ATTENTION=0`. `OLLAMA_FLASH_ATTENTION=1` is safe (same speed, slightly less VRAM);
`OLLAMA_KV_CACHE_TYPE=q8_0` gives no measurable benefit. Tuned tags bake in their own `num_ctx`, so the server default
only matters for untuned models.

## Configuration applied

- **Pi as the native Orca agent** (`~/.pi/agent/settings.json`, `models.json`; snapshots in `config/`; backups
  `*.bak-2026-09-20` next to them): default provider `ollama`, model `ornith-agent:9b`, thinking off, compaction
  reserve 4096 / keep 8000 (fits a 32k window). Orca launches `pi` for `--agent pi`; Pi's Orca extensions
  (`orca-agent-status.ts` etc.) were already installed, and Orca showed the agent as `pi`, state `done`.
- **opencode** (`~/.config/opencode/opencode.json`, backup `opencode.json.bak-2026-09-20`, snapshot `config/opencode.json`):
  default model `ollama/gemma4-agent:12b`, tuned tags with `limit.context/output`, sampling `options`, agent `micro`
  with `maxSteps: 15`, `doom_loop: deny`. The sweep/ablation tags (`sw-*`, `abl-*`) were also registered there.
  Diff:

```diff
--- /home/llucasshenrique/.config/opencode/opencode.json.bak-2026-09-20	2026-09-20 03:01:05.344345304 -0300
+++ /home/llucasshenrique/.config/opencode/opencode.json	2026-09-20 05:21:45.133877339 -0300
@@ -1,7 +1,7 @@
 {
   "$schema": "https://opencode.ai/config.json",
-  "model": "ollama/qwen3:14b",
-  "small_model": "ollama/qwen3:4b",
+  "model": "ollama/gemma4-agent:12b",
+  "small_model": "ollama/gemma4-agent:12b",
   "provider": {
     "ollama": {
       "npm": "@ai-sdk/openai-compatible",
@@ -24,8 +24,196 @@
         },
         "qwen3:14b": {
           "name": "qwen3:14b"
+        },
+        "gemma4-agent:12b": {
+          "name": "gemma4-agent:12b (tuned)",
+          "tool_call": true,
+          "reasoning": false,
+          "limit": {
+            "context": 32768,
+            "output": 4096
+          },
+          "options": {
+            "temperature": 0.2,
+            "top_p": 0.9,
+            "presence_penalty": 0.3,
+            "frequency_penalty": 0.1
+          }
+        },
+        "qwen3-agent:14b": {
+          "name": "qwen3-agent:14b (tuned)",
+          "tool_call": true,
+          "reasoning": true,
+          "limit": {
+            "context": 16384,
+            "output": 4096
+          },
+          "options": {
+            "temperature": 0.25,
+            "top_p": 0.9,
+            "presence_penalty": 0.3,
+            "frequency_penalty": 0.1
+          }
+        },
+        "qwen3-agent:8b": {
+          "name": "qwen3-agent:8b (tuned)",
+          "tool_call": true,
+          "reasoning": true,
+          "limit": {
+            "context": 16384,
+            "output": 4096
+          },
+          "options": {
+            "temperature": 0.25,
+            "top_p": 0.9,
+            "presence_penalty": 0.3,
+            "frequency_penalty": 0.1
+          }
+        },
+        "sw-ornith-9b-q4-k-m": {
+          "name": "sw-ornith-9b-q4-k-m",
+          "tool_call": true,
+          "limit": {
+            "context": 16384,
+            "output": 4096
+          },
+          "options": {
+            "temperature": 0.25,
+            "top_p": 0.9,
+            "presence_penalty": 0.3,
+            "frequency_penalty": 0.1
+          }
+        },
+        "sw-ornith-9b-q8-0": {
+          "name": "sw-ornith-9b-q8-0",
+          "tool_call": true,
+          "limit": {
+            "context": 16384,
+            "output": 4096
+          },
+          "options": {
+            "temperature": 0.25,
+            "top_p": 0.9,
+            "presence_penalty": 0.3,
+            "frequency_penalty": 0.1
+          }
+        },
+        "sw-granite4-1-8b-q3-k-m": {
+          "name": "sw-granite4-1-8b-q3-k-m",
+          "tool_call": true,
+          "limit": {
+            "context": 16384,
+            "output": 4096
+          },
+          "options": {
+            "temperature": 0.25,
+            "top_p": 0.9,
+            "presence_penalty": 0.3,
+            "frequency_penalty": 0.1
+          }
+        },
+        "sw-granite4-1-8b-q4-k-m": {
+          "name": "sw-granite4-1-8b-q4-k-m",
+          "tool_call": true,
+          "limit": {
+            "context": 16384,
+            "output": 4096
+          },
+          "options": {
+            "temperature": 0.25,
+            "top_p": 0.9,
+            "presence_penalty": 0.3,
+            "frequency_penalty": 0.1
+          }
+        },
+        "sw-granite4-1-8b-q6-k": {
+          "name": "sw-granite4-1-8b-q6-k",
+          "tool_call": true,
+          "limit": {
+            "context": 16384,
+            "output": 4096
+          },
+          "options": {
+            "temperature": 0.25,
+            "top_p": 0.9,
+            "presence_penalty": 0.3,
+            "frequency_penalty": 0.1
+          }
+        },
+        "sw-granite4-1-3b-q4-k-m": {
+          "name": "sw-granite4-1-3b-q4-k-m",
+          "tool_call": true,
+          "limit": {
+            "context": 16384,
+            "output": 4096
+          },
+          "options": {
+            "temperature": 0.25,
+            "top_p": 0.9,
+            "presence_penalty": 0.3,
+            "frequency_penalty": 0.1
+          }
+        },
+        "sw-granite4-1-3b-q8-0": {
+          "name": "sw-granite4-1-3b-q8-0",
+          "tool_call": true,
+          "limit": {
+            "context": 16384,
+            "output": 4096
+          },
+          "options": {
+            "temperature": 0.25,
+            "top_p": 0.9,
+            "presence_penalty": 0.3,
+            "frequency_penalty": 0.1
+          }
+        },
+        "sw-qwen3-8b-q8-0": {
+          "name": "sw-qwen3-8b-q8-0",
+          "tool_call": true,
+          "limit": {
+            "context": 16384,
+            "output": 4096
+          },
+          "options": {
+            "temperature": 0.25,
+            "top_p": 0.9,
+            "presence_penalty": 0.3,
+            "frequency_penalty": 0.1
+          }
+        },
+        "abl-gemma4-ctxonly": {
+          "name": "abl-gemma4-ctxonly",
+          "tool_call": true,
+          "limit": {
+            "context": 32768,
+            "output": 4096
+          }
+        },
+        "abl-gemma4-samponly": {
+          "name": "abl-gemma4-samponly",
+          "tool_call": true,
+          "limit": {
+            "context": 32768,
+            "output": 4096
+          }
         }
       }
     }
+  },
+  "agent": {
+    "micro": {
+      "description": "Small single-file, well specified code tasks with a test command",
+      "mode": "primary",
+      "maxSteps": 15,
+      "permission": {
+        "doom_loop": "deny",
+        "edit": "allow",
+        "bash": "allow"
+      }
+    }
+  },
+  "permission": {
+    "doom_loop": "deny"
   }
-}
+}
\ Nenhum caractere de nova linha no final do arquivo
```

## Launching a micro-task

```
# native Orca worker (separate worktree, agent in the first terminal):
orca-ide worktree create --name <task> --no-parent --agent pi --prompt "<task + test command>" --json
# same worktree, new terminal:
orca-ide terminal create --worktree active --command "pi" --json
# headless one-shot (no Orca): 
pi -p "<task>" --no-session
# lightest option: python3 harness/minimal_agent.py ornith-agent:9b "<task>"
```

## Plugging into Orca orchestration

Tested: `worktree create --agent pi --prompt ...` starts Pi in the first terminal with the local model and Orca tracks it
(agent type `pi`, state `done`, last assistant message available in `orca-ide worktree ps`).
Not tested here: the supervised flow `orca-ide terminal create --command "<harness cmd>"` then
`orca-ide orchestration worker-start --terminal <handle> --worktree <sel> --spec ...`. Non-Claude workers cannot send
`worker_done`; have them report with `orca-ide orchestration send --type status`, or poll `worktree ps` for `done`.

## Caveats and open problems

- Small samples: 3 easy tasks, 9 runs per finalist. Differences of a few seconds are noise (e.g. gemma4-agent task 2 took
  59 s once and 152 s another time). These tasks do not test multi-file reasoning or long logs.
- The first opencode matrix predates the fixes below; its untuned `gemma4:12b` 0/3 is explained by the 4096 default context.
- The first sweep attempt and a concurrent stray run produced invalid rows; they are kept in `results/invalid/` and excluded.
  qwen3:8b q8_0 was re-run alone. One 3-second overlap may have touched granite 3B q8_0 (result looked normal).
- The container Ollama that served earlier measurements (31 tok/s) had FA + q8 KV and different settings; all numbers
  here are from the local host service unless stated.
- Pi's `ollama launch` installs `@ollama/pi-web-search`, which adds tool schema overhead (Pi peak prompt ~4-5k tokens).
- `orchestrator:latest`, `qwen3:4b`, `gemma4:latest` existed only in the old container store and were not re-tested.
- qwen-code, codex `--oss`, cline, goose and crush were not evaluated (adapters exist for qwen/codex/cline, untested).
- Pi and Orca defaults were changed globally for Pi; restore with the `*.bak-2026-09-20` files if needed.
