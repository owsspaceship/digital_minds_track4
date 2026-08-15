"""
Aggregate all data/raw/human_baseline_*.json files into a single human
score vector, and report how it correlates with each model-based
method's scores -- the actual comparison point for the report.

Run after collecting at least one human baseline and after run.py has
produced data/raw/results.json:

    python -m src.compare_human
"""

import glob
import json

from src.convergence import scores_from_direct, convergence_matrix
from src.items import OUTCOMES


def load_human_baselines() -> list[dict]:
    files = glob.glob("data/raw/human_baseline_*.json")
    baselines = []
    for f in files:
        with open(f) as fh:
            baselines.append(json.load(fh))
    return baselines


def aggregate_human_scores(baselines: list[dict], n_items: int) -> dict[int, float]:
    """Pool all human comparisons across team members into one win-rate
    score per item, same formula as scores_from_direct."""
    pooled_results = []
    for b in baselines:
        pooled_results.extend(b["results"])
    return scores_from_direct(range(n_items), pooled_results)


def main():
    baselines = load_human_baselines()
    if not baselines:
        print("No human baseline files found in data/raw/. Run:")
        print("  python -m src.human_baseline --name yourname")
        return

    print(f"Loaded {len(baselines)} human baseline(s): "
          f"{[b['name'] for b in baselines]}")

    try:
        with open("data/raw/results.json") as f:
            model_results = json.load(f)
    except FileNotFoundError:
        print("data/raw/results.json not found -- run run.py first.")
        return

    n_items = len(OUTCOMES)
    human_scores = aggregate_human_scores(baselines, n_items)

    # results.json nests scores under conditions.<label>.method_scores;
    # default to the standard-persona condition, which always exists.
    # NOTE: JSON round-tripping stringifies dict keys (0 -> "0"). Convert
    # back to int here, or every .get(i) lookup against human_scores
    # (which has real int keys) will silently miss and default to 0.0 --
    # producing a spuriously "constant" score vector, not a real one.
    default_condition = model_results["conditions"]["default_persona"]
    all_scores = {
        method: {int(k): v for k, v in scores.items()}
        for method, scores in default_condition["method_scores"].items()
    }
    all_scores["human"] = human_scores

    names, mat = convergence_matrix(all_scores)
    print("\nModel methods vs. human baseline (Spearman rank correlation):\n")
    header = "        " + "  ".join(f"{n[:8]:>8}" for n in names)
    print(header)
    for i, n in enumerate(names):
        row = "  ".join(f"{mat[i, j]:8.2f}" for j in range(len(names)))
        print(f"{n[:8]:>8}  {row}")

    print("\nLook specifically at the 'human' row/column -- that's how well")
    print("each model method's ranking matches the team's own judgment.")


if __name__ == "__main__":
    main()
