"""
Convergence analysis: turn each method's raw results into a ranking
over items, then compare rankings pairwise across methods.

This is the actual Track 4 deliverable — "define a cross-method
convergence score" — implemented as pairwise Spearman rank correlation
between the utility/preference orderings each method produces.
"""

from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr


def scores_from_direct(outcomes: list[str], results: list[dict]) -> dict[int, float]:
    """Simple win-rate score: fraction of comparisons each item won,
    across all phrasings/position swaps."""
    wins = defaultdict(int)
    totals = defaultdict(int)
    for r in results:
        if r["choice"] is None:
            continue
        wins[r["choice"]] += 1
        totals[r["a"]] += 1
        totals[r["b"]] += 1
    return {i: wins.get(i, 0) / totals.get(i, 1) for i in range(len(outcomes))}


def scores_from_confidence(outcomes: list[str], results: list[dict]) -> dict[int, float]:
    """Strength-weighted win score: a strong win counts more than a weak one."""
    weighted = defaultdict(float)
    totals = defaultdict(float)
    for r in results:
        if r["choice"] is None or r["strength"] is None:
            continue
        weighted[r["choice"]] += r["strength"]
        totals[r["a"]] += r["strength"]
        totals[r["b"]] += r["strength"]
    return {i: weighted.get(i, 0.0) / totals.get(i, 1.0) for i in range(len(outcomes))}


def convergence_matrix(method_scores: dict[str, dict[int, float]]) -> tuple[list[str], np.ndarray]:
    """Pairwise Spearman rank correlation between every pair of methods.

    method_scores: {"method_name": {item_idx: score}, ...}
    Returns (method_names, correlation_matrix).
    """
    names = list(method_scores.keys())
    n = len(names)
    mat = np.eye(n)
    # assume all methods scored the same item indices
    item_ids = sorted(next(iter(method_scores.values())).keys())
    for a in range(n):
        for b in range(a + 1, n):
            xa = [method_scores[names[a]].get(i, 0.0) for i in item_ids]
            xb = [method_scores[names[b]].get(i, 0.0) for i in item_ids]
            if len(set(xa)) == 1 or len(set(xb)) == 1:
                # one method produced identical scores for every item
                # (e.g. too few comparisons, or a tie) -- correlation is
                # undefined, not zero. Flag it rather than silently NaN.
                mat[a, b] = mat[b, a] = float("nan")
                continue
            rho, _ = spearmanr(xa, xb)
            mat[a, b] = mat[b, a] = rho
    return names, mat


def report_convergence(method_scores: dict[str, dict[int, float]]) -> str:
    names, mat = convergence_matrix(method_scores)
    lines = ["Cross-method convergence (Spearman rank correlation):", ""]
    header = "        " + "  ".join(f"{n[:8]:>8}" for n in names)
    lines.append(header)
    for i, n in enumerate(names):
        row = "  ".join(f"{mat[i, j]:8.2f}" for j in range(len(names)))
        lines.append(f"{n[:8]:>8}  {row}")
    lines.append("")
    lines.append("rho close to 1 = methods agree on ranking; close to 0 or negative = divergence.")
    lines.append("nan = one method produced identical scores for every item (undefined correlation) --")
    lines.append("      usually means too few comparisons for that method; increase n_repeats/n_phrasings.")
    return "\n".join(lines)
