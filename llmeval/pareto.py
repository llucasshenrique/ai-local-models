"""Pareto frontier calculation and recommendation policy for multi-objective model evaluation."""

DEFAULT_OBJECTIVES = {
    "rate": "max",      # Pass rate: maximize
    "wall": "min",      # Wall time latency: minimize
    "size_gb": "min",   # VRAM / footprint: minimize
    "failures": "min"   # Timeouts, loops, errors: minimize
}

def dominates(cand_a, cand_b, objectives=None):
    """
    Returns True if cand_a Pareto-dominates cand_b.
    cand_a dominates cand_b iff:
      - cand_a is at least as good as cand_b on ALL objectives, AND
      - cand_a is strictly better than cand_b on AT LEAST ONE objective.
    """
    objs = objectives or DEFAULT_OBJECTIVES
    at_least_as_good = True
    strictly_better = False

    for metric, direction in objs.items():
        val_a = cand_a.get(metric)
        val_b = cand_b.get(metric)

        # If both are None or equal, neither is better
        if val_a is None and val_b is None:
            continue
        if val_a is None: # Missing in A: A cannot dominate on this metric
            at_least_as_good = False
            break
        if val_b is None: # Present in A, missing in B: A is strictly better
            strictly_better = True
            continue

        if direction == "max":
            if val_a < val_b:
                at_least_as_good = False
                break
            elif val_a > val_b:
                strictly_better = True
        elif direction == "min":
            if val_a > val_b:
                at_least_as_good = False
                break
            elif val_a < val_b:
                strictly_better = True

    return at_least_as_good and strictly_better

def compute_pareto_frontier(candidates, objectives=None):
    """
    Computes the true non-dominated set (Pareto frontier) from a list of candidates.
    Returns: list of non-dominated candidate dicts.
    """
    if not candidates:
        return []
    
    frontier = []
    for cand in candidates:
        is_dominated = False
        for other in candidates:
            if other is cand:
                continue
            if dominates(other, cand, objectives):
                is_dominated = True
                break
        if not is_dominated:
            frontier.append(cand)
            
    return frontier

def select_recommendation(candidates, policy="balanced", tolerance=0.05, objectives=None):
    """
    Applies a distinct, configurable recommendation policy over the candidates and their Pareto frontier.
    Returns: (pick_dict or None, why_str, pareto_frontier)
    """
    have = [c for c in candidates if c.get("rate") is not None]
    if not have:
        return None, "no results yet", []

    frontier = compute_pareto_frontier(have, objectives)

    # Filter to candidates that fit fully on GPU if any do
    fits_gpu = [c for c in have if c.get("gpu_pct") in (None, 100)]
    pool = fits_gpu or have
    pool_frontier = [c for c in frontier if c in pool] or frontier

    if policy == "max_quality":
        # Pure pass-rate optimizer
        pick = max(pool_frontier, key=lambda c: (c.get("rate", 0), -(c.get("wall") or 1e9)))
        why = f"Pareto-optimal with maximum pass rate ({100*pick.get('rate', 0):.0f}%)"
    elif policy == "fastest":
        # Fastest acceptable candidate
        threshold = max(c.get("rate", 0) for c in pool_frontier) - tolerance
        acceptable = [c for c in pool_frontier if c.get("rate", 0) >= threshold]
        pick = min(acceptable or pool_frontier, key=lambda c: (c.get("wall") or 1e9, -(c.get("rate", 0))))
        why = f"Pareto-optimal fastest ({pick.get('wall')}s) within {int(tolerance*100)}pp of top quality"
    elif policy == "balanced":
        # Balanced: top pass rate within tolerance, then smallest footprint, then fastest
        top_rate = max(c.get("rate", 0) for c in pool_frontier)
        near_top = [c for c in pool_frontier if c.get("rate", 0) >= top_rate - tolerance]
        # Sort by size_gb, then wall latency
        pick = sorted(near_top, key=lambda c: (c.get("size_gb") or 999, c.get("wall") or 1e9))[0]
        why = f"Pareto-optimal smallest footprint ({pick.get('size_gb')} GB) within {int(tolerance*100)}pp of top pass rate ({100*top_rate:.0f}%)"
    elif policy == "pareto":
        # No single winner forced; return best among frontier
        pick = pool_frontier[0] if pool_frontier else None
        why = f"Pareto frontier contains {len(pool_frontier)} non-dominated configurations"
    else:
        pick = pool_frontier[0] if pool_frontier else None
        why = f"Selected under policy '{policy}'"

    if not fits_gpu:
        why += "; NOTE: none fit fully on GPU"

    return pick, why, frontier
