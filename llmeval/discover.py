"""Automatic family discovery: find a model family on the ollama library (and HuggingFace when needed), work out what fits
this machine, and let the best local model choose sizes and quantizations given what has already been tested.

Pipeline (each step is inspectable; nothing is downloaded or written unless you apply the plan):
  1. machine()      GPU/RAM facts.
  2. ollama_tags() / hf_candidates()   candidate variants with their download sizes.
  3. tier()         weights + a KV/overhead estimate learned from your own fit measurements -> likely / risky / too big.
  4. choose()       the advisor model picks a few informative variants from ids it is given (validated; deterministic fallback).
  5. apply()        writes a [[families]] block with the explicit tags; prepare/run/tune then work on it as usual."""
import json, math, os, re, subprocess, urllib.parse, urllib.request
from datetime import datetime, timezone
from . import advisor, families as fam, ollama, report, store
from .fit import FIT

UA = {"User-Agent": "llmeval/0.1"}
SKIP = re.compile(r"cloud|mmproj|imatrix", re.I)
QUANT_ORDER = ["q4_K_M", "q6_K", "q8_0", "q5_K_M", "q4_K_S", "q4_0", "q3_K_M", "bf16", "fp16"]

def _get(url, timeout=25):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read().decode("utf-8", "ignore")

def _parse_popularity_num(s):
    if not s: return 0
    s = str(s).strip().lower()
    m = re.search(r"([0-9.]+)\s*([kmb])?", s)
    if not m: return 0
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "k": val *= 1_000
    elif unit == "m": val *= 1_000_000
    elif unit == "b": val *= 1_000_000_000
    return int(val)

def _parse_relative_age_days(s):
    if not s: return None
    m = re.search(r"(\d+)\s*(year|month|week|day|hour|minute)", s.lower())
    if not m: return None
    num = int(m.group(1))
    unit = m.group(2)
    if unit.startswith("year"): return num * 365
    if unit.startswith("month"): return num * 30
    if unit.startswith("week"): return num * 7
    if unit.startswith("day"): return num
    return 0

def machine():
    """{gpu, vram_gb, ram_gb}. VRAM from nvidia-smi (None when unavailable)."""
    m = {"gpu": None, "vram_gb": None, "ram_gb": None}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        name, mib = [x.strip() for x in out.split(",")]; m.update(gpu=name, vram_gb=round(float(mib) / 1024, 1))
    except Exception: pass
    try: m["ram_gb"] = round(int(re.search(r"MemTotal:\s+(\d+)", open("/proc/meminfo").read()).group(1)) / 1048576, 1)
    except Exception: pass
    return m

# ---- sources (parsers are pure functions over the fetched text so they can be tested offline) ----
def parse_ollama_tags(html, name, pulls=None):
    """[{tag, digest, size_gb, source, age, popularity}] from https://ollama.com/library/<name>/tags. Alias tags (same digest as an explicit
    quant tag, e.g. 'latest' or '8b') are dropped in favour of the explicit one."""
    found = {}
    for m in re.finditer(r'href="/library/' + re.escape(name) + r':([A-Za-z0-9._-]+)"', html):
        tag = m.group(1)
        if tag in found or SKIP.search(tag): continue
        chunk = html[m.end():m.end() + 1200]
        d, sz = re.search(r'font-mono">\s*([0-9a-f]{12})', chunk), re.search(r"(\d+(?:\.\d+)?)\s*(GB|MB)", chunk)
        age_m = re.search(r'(\d+\s+(?:year|month|week|day|hour|minute)s?\s+ago)', chunk)
        age_str = age_m.group(1) if age_m else ""
        age_days = _parse_relative_age_days(age_str)
        if sz:
            found[tag] = {
                "digest": d.group(1) if d else tag,
                "size_gb": round(float(sz.group(1)) / (1 if sz.group(2) == "GB" else 1024), 2),
                "age": age_str,
                "age_days": age_days,
            }
    by_digest = {}
    for tag, v in found.items(): by_digest.setdefault(v["digest"], []).append(tag)
    out = []
    pop_str = f"{pulls} pulls" if pulls and "pull" not in str(pulls).lower() else (str(pulls) if pulls else "-")
    pop_num = _parse_popularity_num(pulls) if pulls else 0
    for digest, tags in by_digest.items():
        best = sorted(tags, key=lambda t: (fam.parse_variant(f"{name}:{t}")[1] is None, len(t)))[0]
        v = found[best]
        out.append({
            "tag": f"{name}:{best}",
            "digest": digest,
            "size_gb": v["size_gb"],
            "source": "ollama",
            "age": v.get("age") or "-",
            "age_days": v.get("age_days"),
            "popularity": pop_str,
            "popularity_num": pop_num,
        })
    return out

def parse_hf(repo_id, info, downloads=None, likes=None, created_at=None):
    """GGUF variants of one HuggingFace repo (from /api/models/<id>?blobs=true) as ollama-pullable `hf.co/<id>:<QUANT>` tags."""
    out = {}
    for f in info.get("siblings", []):
        n = f.get("rfilename", "")
        if not n.lower().endswith(".gguf") or SKIP.search(n) or re.search(r"-\d{5}-of-\d{5}", n): continue
        q = fam.QUANT.search(n)
        if q and f.get("size"): out[q.group(1).upper() if q.group(1).lower().startswith("q") else q.group(1)] = round(f["size"] / 1e9, 2)
    pop_str = "-"
    pop_num = downloads or 0
    if downloads is not None:
        if downloads >= 1_000_000: pop_str = f"{downloads/1_000_000:.1f}M dl"
        elif downloads >= 1_000: pop_str = f"{downloads/1_000:.1f}K dl"
        else: pop_str = f"{downloads} dl"
        if likes: pop_str += f", {likes} \u2665"
    age_str = "-"
    age_days = None
    if created_at:
        age_str = str(created_at)[:10]
        try:
            dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
            age_days = max(0, (datetime.now(timezone.utc) - dt).days)
        except Exception:
            pass
    return [{"tag": f"hf.co/{repo_id}:{q}", "digest": f"hf:{repo_id}:{q}", "size_gb": gb, "source": "huggingface",
             "downloads": downloads, "likes": likes, "popularity": pop_str, "popularity_num": pop_num,
             "age": age_str, "age_days": age_days} for q, gb in out.items()]

def ollama_pulls(name):
    """Fetch total pulls for family from ollama.com/search?q=<name> if available."""
    try:
        url = f"https://ollama.com/search?q={urllib.parse.quote(name)}"
        html = _get(url, timeout=6)
        idx = html.find(f"/library/{name}")
        if idx != -1:
            chunk = html[idx:idx + 5000]
            m = re.search(r"<span\s*>\s*([0-9.]+[KMB]?)\s*</span>\s*<span[^>]*>&nbsp;Pulls</span>", chunk)
            if m: return m.group(1)
    except Exception:
        pass
    return None

def ollama_tags(name):
    pulls = ollama_pulls(name)
    return parse_ollama_tags(_get(f"https://ollama.com/library/{urllib.parse.quote(name)}/tags"), name, pulls=pulls)

def hf_candidates(name, repos=4):
    """Top GGUF repos for the family by downloads. HuggingFace has many fine-tunes: the advisor is told the repo owner and popularity."""
    q = urllib.parse.urlencode({"search": name, "filter": "gguf", "sort": "downloads", "direction": "-1", "limit": 12})
    toks = [t for t in re.split(r"[^a-z0-9.]+", name.lower()) if t]
    out = []
    try:
        models = json.loads(_get("https://huggingface.co/api/models?" + q))
    except Exception:
        return []
    for r in models:
        rid = r["id"]
        if not all(t in rid.lower().replace("_", "-") for t in toks): continue
        try: info = json.loads(_get(f"https://huggingface.co/api/models/{rid}?blobs=true"))
        except Exception: continue
        out += parse_hf(rid, info, downloads=r.get("downloads"), likes=r.get("likes"), created_at=r.get("createdAt") or r.get("lastModified"))
        if len({c["tag"].split(":")[0] for c in out}) >= repos: break
    return out

# ---- feasibility, learned from your own measurements ----
def overhead_gb(ctx):
    """KV/compute overhead beyond the weights at `ctx`, taken from the worst case you measured (fit.jsonl vs installed sizes);
    without data, a conservative default scaled from ctx. It is an estimate: measure with `llmeval fit` after pulling."""
    try: inst = {m["name"]: m["size"] / 1e9 for m in ollama.api("/api/tags")["models"]}
    except Exception: inst = {}
    worst = 0.0
    for r in store.rows(FIT):
        f = r.get("fit"); w = inst.get(r["model"] if ":" in r["model"] else r["model"] + ":latest")
        if f and w and r["num_ctx"] >= ctx * 0.9: worst = max(worst, f["size_gb"] - w)
    return round(worst, 1) if worst else round(2.6 * ctx / 16384, 1)

def tier(cands, mach, ctx, reserve=0.6):
    """Annotate each candidate: likely (weights + overhead fit), risky (weights alone fit), too_big."""
    vram = mach.get("vram_gb") or 8.0; oh = overhead_gb(ctx)
    for c in cands:
        c["size"], c["quant"] = fam.parse_variant(c["tag"])
        c["est_total_gb"] = round(c["size_gb"] + oh, 1)
        c["tier"] = "likely" if c["est_total_gb"] <= vram - reserve else "risky" if c["size_gb"] <= vram - 1.0 else "too_big"
    return [c for c in cands if c["tier"] != "too_big"], oh

def candidate_score(c):
    """Multi-feature scoring used to rank variants for discovery and selection.
    Combines feasibility tier, parameter capacity, quantization quality, popularity, and age/freshness."""
    tier = c.get("tier", "risky")
    tier_score = 150.0 if tier == "likely" else 30.0 if tier == "risky" else -100.0
    params = fam.params_b(c.get("size")) or 1.0
    size_score = min(params, 70.0) * 2.5
    q = c.get("quant") or ""
    q_idx = QUANT_ORDER.index(q) if q in QUANT_ORDER else 99
    quant_score = max(0.0, 35.0 - q_idx * 3.5)
    pop_num = c.get("popularity_num") or 0
    pop_score = min(25.0, 3.0 * math.log10(pop_num + 1)) if pop_num > 0 else 0.0
    age_days = c.get("age_days")
    if age_days is not None:
        if age_days <= 60: age_score = 15.0
        elif age_days <= 180: age_score = 10.0
        elif age_days <= 365: age_score = 5.0
        else: age_score = 1.0
    else:
        age_score = 4.0
    source_score = 5.0 if c.get("source") == "ollama" else 0.0
    return tier_score + size_score + quant_score + pop_score + age_score + source_score

def heuristic(cands, k):
    """No-LLM fallback: highest scoring variant, its neighbour in quant, and one size down."""
    pool = sorted([c for c in cands if c.get("tier") == "likely"] or cands, key=lambda c: -candidate_score(c))
    if not pool: return []
    top = pool[0]; pick = [top] + [c for c in pool if c.get("size") == top.get("size") and c is not top][:1] + [c for c in pool if c.get("size") != top.get("size")][:1]
    return pick[:k]

def format_top_candidates(cands):
    """Format top candidate models with all features used for selection."""
    if not cands: return []
    lines = [
        "--- Top 10 Candidate Models for Selection ---",
        f"{'#':<3} {'Tag':<42} {'Params':<7} {'Quant':<8} {'Size':>7} {'Est.RAM':>8} {'Tier':<7} {'Popularity':<14} {'Age':<12} {'Source':<7}",
        "-" * 115,
    ]
    for i, c in enumerate(cands[:10], 1):
        tag = c.get("tag", "")
        if len(tag) > 40: tag = tag[:37] + "..."
        params = c.get("size") or "?"
        quant = c.get("quant") or "?"
        sz = f"{c.get('size_gb', 0):.1f} GB"
        ram = f"{c.get('est_total_gb', 0):.1f} GB" if c.get("est_total_gb") else "-"
        tier = c.get("tier") or "-"
        pop = c.get("popularity") or "-"
        if len(pop) > 13: pop = pop[:11] + ".."
        age = c.get("age") or "-"
        if len(age) > 11: age = age[:9] + ".."
        src = c.get("source") or "ollama"
        lines.append(f"#{i:<2} {tag:<42} {params:<7} {quant:<8} {sz:>7} {ram:>8} {tier:<7} {pop:<14} {age:<12} {src:<7}")
    lines.append("-" * 115)
    return lines

PROMPT = """You are choosing which variants of the model family "{name}" to evaluate as a local coding-agent worker.
MACHINE: {machine}. Target context: {ctx} tokens. Estimated KV/compute overhead at that context: {oh} GB (learned from this machine).
ALREADY TESTED ON THIS MACHINE (pass rate, speed, VRAM fit):
{known}

CANDIDATES (id | tag | params | quant | download GB | est. total GB | tier | popularity | age | source):
{cands}

Choose at most {k} candidates that together answer "which size and quantization is best here?": vary size and quantization
informatively, skip near-duplicates, avoid very low quantizations (q2/q3) unless nothing else fits, prefer official/popular sources and recent releases,
and prefer 'likely' over 'risky'. Answer with JSON only:
{{"choices": [{{"id": 0, "why": "<one sentence>"}}], "notes": "<what to watch for>"}}"""

def choose(cfg, name, cands, mach, ctx, oh, k=3, model=None, log=print):
    """(chosen candidates with reasons, notes, advisor tag or None when the fallback was used)."""
    idx = {i: c for i, c in enumerate(cands)}
    try:
        tag, why = advisor.pick_advisor(cfg, model)
        known = "\n".join(f'- {x["model"]} via {x["harness"]}: {x["passed"]}/{x["n"]}, median {x["wall"]}s' for x in report.summary()[:10]) or "(nothing yet)"
        for fname, rows, pick, w in report.family_table():
            known += f"\n- family {fname}: recommended {pick['tag'] if pick else '-'}"
        table = "\n".join(f'{i} | {c["tag"]} | {c["size"] or "?"} | {c["quant"] or "?"} | {c["size_gb"]} | {c["est_total_gb"]} | {c["tier"]} | {c.get("popularity", "-")} | {c.get("age", "-")} | {c["source"]}'
                          + (f' ({c["downloads"]} downloads)' if c.get("downloads") else "") for i, c in idx.items())
        from .lock import gpu_lock
        with gpu_lock():
            ollama.unload_all()
            try: raw = advisor.ask(tag, PROMPT.format(name=name, machine=json.dumps(mach), ctx=ctx, oh=oh, known=known, cands=table, k=k))
            finally: ollama.stop(tag)
        data = advisor.parse_json(raw); picked, seen = [], set()
        for ch in data.get("choices", []):
            i = ch.get("id"); c = idx.get(int(i)) if str(i).lstrip("-").isdigit() else None
            if c and c["tag"] not in seen: seen.add(c["tag"]); picked.append({**c, "why": str(ch.get("why", ""))[:200]})
        if picked: return picked[:k], str(data.get("notes", ""))[:300], tag
        log("advisor returned no valid choices: using the heuristic")
    except Exception as e:
        log(f"advisor unavailable ({str(e)[:80]}): using the heuristic")
    return [{**c, "why": "heuristic: highest scoring candidate, neighbouring quantization, one size down"} for c in heuristic(cands, k)], "", None

def discover(cfg, name, ctx=32768, k=3, use_hf=False, model=None, log=print):
    """Full plan for one family. Returns a dict; changes nothing on disk."""
    mach = machine(); cands = []
    try: cands = ollama_tags(name); log(f"ollama library: {len(cands)} variants")
    except Exception as e: log(f"ollama library lookup failed: {str(e)[:80]}")
    if use_hf or len(cands) < 2:
        try: hf = hf_candidates(name); log(f"HuggingFace: {len(hf)} GGUF variants"); cands += hf
        except Exception as e: log(f"HuggingFace lookup failed: {str(e)[:80]}")
    if not cands: raise SystemExit(f"no variants found for '{name}' (offline, or the name is wrong)")
    ok, oh = tier(cands, mach, ctx)
    if not ok: raise SystemExit(f"nothing fits this machine ({mach.get('vram_gb')} GB VRAM) at {ctx} context")
    ok.sort(key=lambda c: -candidate_score(c))
    top_10 = ok[:10]
    log(f"--- Evaluated {len(ok)} feasible candidate variants for '{name}' ---")
    for line in format_top_candidates(top_10):
        log(line)
    chosen, notes, adv = choose(cfg, name, ok, mach, ctx, oh, k, model, log)
    inst = ollama.installed()
    return {"name": name, "ctx": ctx, "machine": mach, "overhead_gb": oh, "candidates": len(cands), "feasible": len(ok),
            "top_candidates": top_10, "chosen": chosen, "notes": notes,
            "advisor": adv, "download_gb": round(sum(c["size_gb"] for c in chosen if c["tag"] not in inst and c["tag"] + ":latest" not in inst), 1)}

def format_plan(p):
    m = p["machine"]; out = [f'{p["name"]}: {p["candidates"]} variants found, {p["feasible"]} plausible on {m.get("gpu")} ({m.get("vram_gb")} GB VRAM, {m.get("ram_gb")} GB RAM) at {p["ctx"]} context',
                             f'estimated overhead beyond weights: {p["overhead_gb"]} GB; chosen by: {p["advisor"] or "heuristic (no advisor)"}']
    if p.get("top_candidates"):
        out.extend(format_top_candidates(p["top_candidates"]))
    out += [f'  + {c["tag"]:46} {c["size"] or "?":>5} {c["quant"] or "?":>7} {c["size_gb"]:>5} GB  [{c["tier"]}, {c["source"]}]  {c["why"]}' for c in p["chosen"]]
    if p["notes"]: out.append("notes: " + p["notes"])
    return out + [f'download needed: {p["download_gb"]} GB (models already installed are not counted)']

def apply(path, plan, log=print):
    """Write the plan as a [[families]] block with explicit variants. Nothing is downloaded here."""
    from . import config
    if any(f["name"] == plan["name"] for f in config.load(path).get("families", [])): log(f'skip: family {plan["name"]} already in the config'); return False
    q = ", ".join(f'"{c["tag"]}"' for c in plan["chosen"])
    open(path, "a").write(f'\n# discovered by llmeval ({plan["advisor"] or "heuristic"}); {plan["download_gb"]} GB to download\n[[families]]\nname     = "{plan["name"]}"\nvariants = [{q}]\nctx      = {plan["ctx"]}\n')
    log(f'added family {plan["name"]}: {len(plan["chosen"])} variants'); return True
