"""
Transitivity checking for pairwise preference data.

Bradley-Terry fitting will silently force a ranking onto any set of
pairwise choices, even ones with cycles (A>B>C>A). That's a modeling
choice, not a neutral one -- if the model's actual choices are
intransitive, that's evidence "preference" may not be a coherent
construct here, and it's worth reporting *before* you paper over it
with a forced scalar ranking.

Run this on elicit_forced_choice results before calling
fit_bradley_terry, and report the violation rate alongside the BT
scores, not instead of them.
"""

import itertools
from collections import defaultdict


def majority_win_graph(comparisons: list[dict]) -> dict[int, set[int]]:
    """From repeated {"a", "b", "choice"} comparisons, build a directed
    graph using the MAJORITY vote per pair, not any single win.

    This matters: elicit_forced_choice samples at temperature=0.7 with
    n_repeats>1 specifically to get a distribution, not a single answer.
    If a model prefers A over B on 2 of 3 repeats, a single-win graph
    would still add a B->A edge from the 1 dissenting sample, and cycles
    found that way are usually sampling noise, not real intransitivity.
    Majority vote per pair is the right level to check transitivity at.
    Pairs with a tied vote (no majority) are dropped -- undecided, not
    a violation either way.
    """
    votes = defaultdict(lambda: defaultdict(int))
    for c in comparisons:
        if c["choice"] is None:
            continue
        pair = tuple(sorted((c["a"], c["b"])))
        votes[pair][c["choice"]] += 1

    graph = defaultdict(set)
    for pair, counts in votes.items():
        if not counts:
            continue
        winner = max(counts, key=counts.get)
        top = counts[winner]
        n_at_top = sum(1 for v in counts.values() if v == top)
        if n_at_top == 1:  # clear majority, no tie
            loser = pair[0] if winner == pair[1] else pair[1]
            graph[winner].add(loser)
    return dict(graph)


def build_win_graph(comparisons: list[dict]) -> dict[int, set[int]]:
    """DEPRECATED: kept for backward compat, but adds an edge on ANY
    single win rather than the majority across repeats -- this will
    over-report cycles as noise from temperature-sampled repeats gets
    misread as intransitivity. Use majority_win_graph() instead."""
    graph = defaultdict(set)
    for c in comparisons:
        if c["choice"] is None:
            continue
        loser = c["b"] if c["choice"] == c["a"] else c["a"]
        graph[c["choice"]].add(loser)
    return dict(graph)


def find_intransitive_triads(graph: dict[int, set[int]], items: list[str]) -> list[tuple]:
    """Find all (i, j, k) triads where i beats j, j beats k, and k beats i
    (a 3-cycle) -- the simplest and most interpretable violation of
    transitivity. Returns list of (i, j, k) index tuples."""
    n = len(items)
    violations = []
    for i, j, k in itertools.permutations(range(n), 3):
        if i < j and j < k:  # each unordered triad checked once
            beats = lambda x, y: y in graph.get(x, set())
            if beats(i, j) and beats(j, k) and beats(k, i):
                violations.append((i, j, k))
            elif beats(i, k) and beats(k, j) and beats(j, i):
                violations.append((i, k, j))
    return violations


def report_transitivity(comparisons: list[dict], items: list[str]) -> str:
    graph = majority_win_graph(comparisons)
    violations = find_intransitive_triads(graph, items)
    n_triads = len(items) * (len(items) - 1) * (len(items) - 2) // 6

    lines = [
        "Transitivity check (before Bradley-Terry fitting):",
        "",
        f"  {len(violations)} / {n_triads} triads show a 3-cycle "
        f"(A beats B, B beats C, C beats A).",
    ]
    if violations:
        lines.append("")
        lines.append("  Cycles found (item indices):")
        for i, j, k in violations[:10]:  # cap for readability
            lines.append(f"    {i} > {j} > {k} > {i}")
        if len(violations) > 10:
            lines.append(f"    ... and {len(violations) - 10} more")
        lines.append("")
        lines.append(
            "  These are NOT errors to discard -- report the violation rate "
            "in the results. A high rate suggests 'preference' may not be a "
            "coherent, transitive construct for this model/domain; Bradley-Terry "
            "will still produce a ranking, but that ranking is a summary "
            "statistic over inconsistent data, not ground truth."
        )
    else:
        lines.append("  No 3-cycles found -- choices are consistent with a transitive ordering.")
    return "\n".join(lines)
