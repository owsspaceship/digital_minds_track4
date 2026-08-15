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

    from src.items import OUTCOMES
    from src.elicit import elicit_direct, elicit_forced_choice, fit_bradley_terry, elicit_confidence
    from src.convergence import scores_from_direct, scores_from_confidence, report_convergence

    small = OUTCOMES[:4]

    d = elicit_direct(small, n_phrasings=1)
    assert len(d) > 0, "elicit_direct produced no results"

    fc = elicit_forced_choice(small, n_repeats=2)
    bt = fit_bradley_terry(small, fc)
    assert len(bt) == len(small), "Bradley-Terry scores don't cover all items"

    c = elicit_confidence(small)
    assert len(c) > 0, "elicit_confidence produced no results"

    ms = {
        "direct": scores_from_direct(small, d),
        "forced_choice": bt,
        "confidence": scores_from_confidence(small, c),
    }
    print(report_convergence(ms))
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    run_smoke_test()
