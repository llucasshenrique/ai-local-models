"""Statistical analysis for llmeval: confidence intervals, comparison tests, and evidence-based decisions."""
import math
import statistics as st

def wilson_score_interval(successes, total, confidence=0.95):
    """
    Computes the Wilson score interval for a binomial proportion.
    Returns (center, lower, upper).
    Handles edge cases gracefully (total=0 -> (0.0, 0.0, 0.0)).
    """
    if total <= 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    # z-score: 1.96 for 95%, 2.576 for 99%, 1.645 for 90%
    z = 1.959963984540054 if confidence == 0.95 else 2.5758293035489004 if confidence == 0.99 else 1.6448536269514722
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    half_width = (z * math.sqrt((p * (1.0 - p) / total) + (z2 / (4.0 * total * total)))) / denominator
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return round(p, 4), round(lower, 4), round(upper, 4)

def numeric_summary(values):
    """Statistical summary of continuous variables: sample size, mean, median, stdev, min, max, IQR."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None, "iqr": None}
    n = len(clean)
    mean_val = sum(clean) / n
    med_val = st.median(clean)
    std_val = st.stdev(clean) if n >= 2 else 0.0
    s_clean = sorted(clean)
    q1 = s_clean[n // 4]
    q3 = s_clean[(3 * n) // 4]
    iqr_val = q3 - q1
    return {
        "n": n,
        "mean": round(mean_val, 2),
        "median": round(med_val, 2),
        "std": round(std_val, 2),
        "min": round(min(clean), 2),
        "max": round(max(clean), 2),
        "iqr": round(iqr_val, 2)
    }

def fisher_exact_pvalue(k1, n1, k2, n2):
    """
    Computes two-tailed p-value for 2x2 contingency table using hypergeom distribution.
    Table:
       k1     k2
     n1-k1  n2-k2
    Pure Python implementation using math.comb.
    """
    f1 = n1 - k1
    f2 = n2 - k2
    total_k = k1 + k2
    total_f = f1 + f2
    total_n = n1 + n2

    if total_n == 0:
        return 1.0

    def prob(k):
        f = total_k - k
        if 0 <= k <= n1 and 0 <= f <= n2:
            return (math.comb(n1, k) * math.comb(n2, f)) / math.comb(total_n, total_k)
        return 0.0

    observed_prob = prob(k1)
    min_k = max(0, total_k - n2)
    max_k = min(n1, total_k)
    p_val = sum(prob(k) for k in range(min_k, max_k + 1) if prob(k) <= observed_prob + 1e-12)
    return min(1.0, max(0.0, round(p_val, 4)))

def difference_confidence_interval(k_a, n_a, k_b, n_b, confidence=0.95):
    """
    Newcombe-Wilson method for confidence interval of difference between two independent proportions (p_b - p_a).
    Returns (delta, lower, upper).
    """
    if n_a == 0 or n_b == 0:
        return 0.0, 0.0, 0.0
    p_a, l_a, u_a = wilson_score_interval(k_a, n_a, confidence)
    p_b, l_b, u_b = wilson_score_interval(k_b, n_b, confidence)
    delta = p_b - p_a
    lower = delta - math.sqrt((p_b - l_b)**2 + (u_a - p_a)**2)
    upper = delta + math.sqrt((u_b - p_b)**2 + (p_a - l_a)**2)
    return round(delta, 4), round(lower, 4), round(upper, 4)

def compare_runs(records_a, records_b, max_slowdown=0.25, min_reps_for_significance=3):
    """
    Compares two sets of trials with statistical evidence.
    Returns:
    {
      "decision": "improvement_supported" | "no_improvement" | "insufficient_evidence" | "incomparable",
      "observed": {...},
      "statistical_evidence": {...},
      "resource_behavior": {...}
    }
    """
    if not records_a or not records_b:
        return {
            "decision": "insufficient_evidence",
            "reason": "Missing records for one or both variants",
            "observed": {},
            "statistical_evidence": {},
            "resource_behavior": {}
        }

    # Match common tasks
    tasks_a = {r["task"] for r in records_a}
    tasks_b = {r["task"] for r in records_b}
    common_tasks = tasks_a & tasks_b
    if not common_tasks:
        return {
            "decision": "incomparable",
            "reason": "No overlapping tasks between configurations",
            "observed": {},
            "statistical_evidence": {},
            "resource_behavior": {}
        }

    subset_a = [r for r in records_a if r["task"] in common_tasks]
    subset_b = [r for r in records_b if r["task"] in common_tasks]

    n_a = len(subset_a)
    n_b = len(subset_b)
    k_a = sum(1 for r in subset_a if r.get("done"))
    k_b = sum(1 for r in subset_b if r.get("done"))

    p_a, l_a, u_a = wilson_score_interval(k_a, n_a)
    p_b, l_b, u_b = wilson_score_interval(k_b, n_b)
    delta, delta_low, delta_high = difference_confidence_interval(k_a, n_a, k_b, n_b)
    p_value = fisher_exact_pvalue(k_a, n_a, k_b, n_b)

    lat_a = numeric_summary([r.get("wall_s") for r in subset_a])
    lat_b = numeric_summary([r.get("wall_s") for r in subset_b])

    fails_a = sum(1 for r in subset_a if not r.get("done") or r.get("timeout") or r.get("loop"))
    fails_b = sum(1 for r in subset_b if not r.get("done") or r.get("timeout") or r.get("loop"))

    # Latency ratio
    med_lat_a = lat_a["median"] or 1e-9
    med_lat_b = lat_b["median"] or 1e-9
    lat_ratio = round(med_lat_b / med_lat_a, 2) if med_lat_a > 0 else 1.0

    # Decision logic
    # 1. Check if sample size is sufficient for statistical confidence
    reps_per_task_a = n_a / max(1, len(common_tasks))
    reps_per_task_b = n_b / max(1, len(common_tasks))

    has_sufficient_evidence = (min(reps_per_task_a, reps_per_task_b) >= min_reps_for_significance) or (min(n_a, n_b) >= 8)
    if not has_sufficient_evidence or (n_a < 6 or n_b < 6):
        # Small sample size: cannot claim statistically confirmed improvement
        if k_b > k_a:
            decision = "insufficient_evidence"
            decision_reason = f"Candidate passed {k_b}/{n_b} vs {k_a}/{n_a}, but sample size is too small (n={n_b}) to confirm significance"
        elif k_b < k_a:
            decision = "insufficient_evidence"
            decision_reason = f"Candidate passed {k_b}/{n_b} vs {k_a}/{n_a}, trend is worse but sample size is small"
        else:
            decision = "insufficient_evidence"
            decision_reason = f"Equal pass counts ({k_b}/{n_b} vs {k_a}/{n_a}) with small sample size"
    elif delta_low > 0 or (p_value < 0.05 and delta > 0):
        # Statistically significant pass rate gain
        if lat_ratio > (1.0 + max_slowdown):
            decision = "insufficient_evidence"
            decision_reason = f"Pass rate improvement is statistically significant (+{int(delta*100)}pp, p={p_value}), but latency slowed down by {int((lat_ratio-1)*100)}% (exceeds {int(max_slowdown*100)}% threshold)"
        else:
            decision = "improvement_supported"
            decision_reason = f"Pass rate improvement is statistically supported (+{int(delta*100)}pp, 95% CI [{int(delta_low*100)}pp, {int(delta_high*100)}pp], p={p_value})"
    elif delta_high < 0 or (p_value < 0.05 and delta < 0):
        decision = "no_improvement"
        decision_reason = f"Statistically significant regression observed ({int(delta*100)}pp, p={p_value})"
    elif delta == 0 and lat_ratio <= (1.0 - 0.20) and p_value >= 0.5:
        # Same pass rate with large, clean speedup
        decision = "improvement_supported"
        decision_reason = f"Pass rate maintained ({k_b}/{n_b}) with {int((1-lat_ratio)*100)}% latency improvement"
    elif delta < 0 or lat_ratio > (1.0 + max_slowdown):
        decision = "no_improvement"
        decision_reason = f"Observed pass rate did not improve ({k_b}/{n_b} vs {k_a}/{n_a}) or latency exceeded budget"
    else:
        decision = "insufficient_evidence"
        decision_reason = f"Observed difference ({int(delta*100)}pp, 95% CI [{int(delta_low*100)}pp, {int(delta_high*100)}pp], p={p_value}) is not statistically distinguishable from noise"

    return {
        "decision": decision,
        "reason": decision_reason,
        "observed": {
            "a": {"passed": k_a, "total": n_a, "rate": p_a},
            "b": {"passed": k_b, "total": n_b, "rate": p_b},
            "tasks_evaluated": sorted(list(common_tasks))
        },
        "statistical_evidence": {
            "delta_rate": delta,
            "delta_ci_95": [delta_low, delta_high],
            "ci_95_a": [l_a, u_a],
            "ci_95_b": [l_b, u_b],
            "p_value": p_value,
            "min_reps_required": min_reps_for_significance
        },
        "resource_behavior": {
            "wall_a": lat_a,
            "wall_b": lat_b,
            "latency_ratio": lat_ratio,
            "failures_a": fails_a,
            "failures_b": fails_b
        }
    }
