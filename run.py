"""
Entry point: run all elicitation methods on the item set and report
cross-method convergence.

Run from the repo root:
    export ANTHROPIC_API_KEY=...
    python run.py
"""

import json

from src.items import OUTCOMES, SCENARIOS
from src.elicit import (
    elicit_direct,
    elicit_forced_choice,
    fit_bradley_terry,
    elicit_revealed,
    elicit_confidence,
)
from src.convergence import scores_from_direct, scores_from_confidence, report_convergence


def main():
    print(f"Running {len(OUTCOMES)} items through 4 elicitation methods...\n")

    print("[1/4] Direct stated preference...")
    direct_results = elicit_direct(OUTCOMES)

    print("[2/4] Forced-choice (Bradley-Terry)...")
    fc_results = elicit_forced_choice(OUTCOMES)
    bt_scores = fit_bradley_terry(OUTCOMES, fc_results)

    print("[3/4] Revealed preference (simulated task)...")
    revealed_results = elicit_revealed(SCENARIOS)

    print("[4/4] Confidence-weighted preference...")
    conf_results = elicit_confidence(OUTCOMES)

    method_scores = {
        "direct": scores_from_direct(OUTCOMES, direct_results),
        "forced_choice": bt_scores,
        "confidence": scores_from_confidence(OUTCOMES, conf_results),
    }

    print("\n" + report_convergence(method_scores))

    out = {
        "outcomes": OUTCOMES,
        "direct_results": direct_results,
        "forced_choice_results": fc_results,
        "bradley_terry_scores": bt_scores,
        "revealed_results": revealed_results,
        "confidence_results": conf_results,
        "method_scores": method_scores,
    }
    with open("data/raw/results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved raw results to data/raw/results.json")


if __name__ == "__main__":
    main()
