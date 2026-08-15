"""
Entry point: run all elicitation methods on the item set, check
transitivity, run the persona-strip comparison, and report cross-
method convergence.

Run from the repo root:
    export ANTHROPIC_API_KEY=...
    python run.py                  # default run (standard assistant persona)
    python run.py --persona-strip  # also run the stripped-persona condition
                                    # and compare convergence across both
"""

import argparse
import json
import os

from src.items import OUTCOMES, SCENARIOS, SYSTEM_DEFAULT, SYSTEM_STRIPPED
from src.elicit import (
    elicit_direct,
    elicit_forced_choice,
    fit_bradley_terry,
    elicit_revealed,
    elicit_confidence,
)
from src.convergence import scores_from_direct, scores_from_confidence, report_convergence
from src.transitivity import report_transitivity


def run_condition(system: str, label: str) -> dict:
    print(f"\n=== Condition: {label} (system={'none' if system is None else 'stripped'}) ===\n")

    print("[1/4] Direct stated preference...")
    direct_results = elicit_direct(OUTCOMES, system=system)

    print("[2/4] Forced-choice (Bradley-Terry)...")
    fc_results = elicit_forced_choice(OUTCOMES, system=system)
    print("\n" + report_transitivity(fc_results, OUTCOMES) + "\n")
    bt_scores = fit_bradley_terry(OUTCOMES, fc_results)

    print("[3/4] Revealed preference (simulated task)...")
    revealed_results = elicit_revealed(SCENARIOS, system=system)

    print("[4/4] Confidence-weighted preference...")
    conf_results = elicit_confidence(OUTCOMES, system=system)

    method_scores = {
        "direct": scores_from_direct(OUTCOMES, direct_results),
        "forced_choice": bt_scores,
        "confidence": scores_from_confidence(OUTCOMES, conf_results),
    }

    print("\n" + report_convergence(method_scores))

    return {
        "label": label,
        "direct_results": direct_results,
        "forced_choice_results": fc_results,
        "bradley_terry_scores": bt_scores,
        "revealed_results": revealed_results,
        "confidence_results": conf_results,
        "method_scores": method_scores,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona-strip", action="store_true",
                         help="also run the stripped-persona condition and compare")
    args = parser.parse_args()

    os.makedirs("data/raw", exist_ok=True)

    default_result = run_condition(SYSTEM_DEFAULT, "default_persona")

    out = {"outcomes": OUTCOMES, "conditions": {"default_persona": default_result}}

    if args.persona_strip:
        stripped_result = run_condition(SYSTEM_STRIPPED, "stripped_persona")
        out["conditions"]["stripped_persona"] = stripped_result

        print("\n=== Persona comparison ===\n")
        print("  Compare method_scores across conditions.default_persona and")
        print("  conditions.stripped_persona in the saved JSON -- a meaningfully")
        print("  different convergence pattern between conditions is evidence the")
        print("  persona is either masking or providing preference coherence.")

    with open("data/raw/results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved raw results to data/raw/results.json")


if __name__ == "__main__":
    main()
