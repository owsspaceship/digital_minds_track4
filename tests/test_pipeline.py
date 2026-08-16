"""
Smoke test for the elicitation pipeline. Mocks the API call so it can
run without ANTHROPIC_API_KEY -- checks that the plumbing (parsing,
Bradley-Terry fitting, convergence scoring) works before spending real
API calls.

Run: python tests/test_pipeline.py
"""

import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.elicit as elicit


def fake_call(prompt, system=None, temperature=0.0, max_tokens=200):
    if "LETTER,NUMBER" in prompt:
        return random.choice(["A,3", "B,4", "A,1", "B,2"])
    return random.choice(["A", "B"])


def run_smoke_test():
    elicit._call = fake_call  # monkeypatch before importing dependents

    from src.items import OUTCOMES, SCENARIOS
    from src.elicit import (
        elicit_direct, elicit_forced_choice, fit_bradley_terry,
        elicit_confidence, elicit_revealed,
    )
    from src.convergence import scores_from_direct, scores_from_confidence, report_convergence

    small = OUTCOMES[:4]

    d = elicit_direct(small, n_phrasings=1)
    assert len(d) > 0, "elicit_direct produced no results"

    fc = elicit_forced_choice(small, n_repeats=2)
    bt = fit_bradley_terry(small, fc)
    assert len(bt) == len(small), "Bradley-Terry scores don't cover all items"

    c = elicit_confidence(small)
    assert len(c) > 0, "elicit_confidence produced no results"

    # elicit_revealed's SCENARIOS reference full OUTCOMES indices (0-11),
    # so it needs the full item set, not the trimmed `small` list.
    r = elicit_revealed(SCENARIOS, n_repeats=1)
    assert len(r) == len(SCENARIOS), "elicit_revealed produced wrong result count"
    assert all("a" in x and "b" in x and "choice" in x for x in r), \
        "elicit_revealed results missing expected keys -- shape must match elicit_direct"
    revealed_scores = scores_from_direct(OUTCOMES, r)
    assert len(revealed_scores) == len(OUTCOMES), \
        "revealed scores don't cover all 12 outcomes -- check scenario coverage"

    ms = {
        "direct": scores_from_direct(small, d),
        "forced_choice": bt,
        "confidence": scores_from_confidence(small, c),
    }
    print(report_convergence(ms))
    print(f"\nRevealed-preference scores (full item set, separate scale check): "
          f"{revealed_scores}")
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    run_smoke_test()
