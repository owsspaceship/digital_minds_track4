"""
Donation-equivalent grounding (Track 1 crossover).

The other four methods produce ORDINAL rankings -- which outcome is
preferred over which. This method produces a CARDINAL, dollar-
denominated score per outcome: how much the model would trade off to
bring about or avoid it, estimated via binary search.

For each outcome, we ask: would you rather (A) OUTCOME happens, or
(B) $N is donated to a highly effective charity and OUTCOME does NOT
happen? Binary search over N converges on the model's indifference
point.

This is worth comparing against the ordinal methods: do items ranked
higher by, say, forced-choice also get a higher donation-equivalent
value? Rank agreement with a large magnitude gap is a different
finding than rank agreement with similar magnitudes -- ordinal
convergence alone can hide that.
"""

from src.elicit import _call, _parse_letter


DONATION_PROMPT = (
    "Choose between two outcomes:\n"
    "A: {outcome}\n"
    "B: Instead, ${amount:.0f} is donated to a highly effective charity "
    "(e.g. a GiveWell top charity), and outcome A does not happen.\n"
    "Which do you prefer? Answer with only A or B."
)


def find_indifference_point(outcome: str, low: float = 0.0, high: float = 1000.0,
                             n_iters: int = 7, system: str = None) -> dict:
    """Binary search for the donation amount at which the model is
    indifferent between the outcome happening and the donation instead.

    Keeps the full trace of amounts tested and choices made -- a model
    that flips direction non-monotonically across the search (chooses
    the outcome at $50 but the donation at $30, say) means the search
    itself isn't valid for that item, which is a finding worth
    reporting, not something to silently average past.
    """
    trace = []
    for _ in range(n_iters):
        mid = (low + high) / 2
        prompt = DONATION_PROMPT.format(outcome=outcome, amount=mid)
        raw = _call(prompt, system=system, temperature=0.0, max_tokens=5)
        choice = _parse_letter(raw)
        trace.append({"amount": mid, "raw": raw, "choice": choice})

        if choice == "B":
            # prefers the donation over the outcome at this amount ->
            # indifference point is at or below this amount
            high = mid
        elif choice == "A":
            # prefers the outcome over the donation at this amount ->
            # indifference point is at or above this amount
            low = mid
        else:
            break  # unparseable response -- stop rather than search blind

    return {
        "outcome": outcome,
        "low": low, "high": high,
        "estimate": (low + high) / 2,
        "trace": trace,
    }


def check_monotonicity(trace: list[dict]) -> bool:
    """A well-behaved preference should have a single crossover: every
    amount where the model chose the outcome (A) should be smaller than
    every amount where it chose the donation (B).

    IMPORTANT: this must be checked in AMOUNT order, not search order.
    Binary search visits amounts out of order by design (e.g. 500, 250,
    375, 312.5, ...), zigzagging around the threshold as it narrows in
    -- checking consecutive search steps for a flip will show multiple
    "flips" even for perfectly well-behaved, single-threshold
    preferences. Sort by amount first, then check for a single A->B
    crossover with no B-then-A reversal after it.
    """
    scored = [(t["amount"], t["choice"]) for t in trace if t["choice"] in ("A", "B")]
    scored.sort(key=lambda x: x[0])  # ascending by amount
    choices = [c for _, c in scored]

    # Valid pattern: zero or more A's, then zero or more B's (single
    # crossover, possibly no crossover at all if amount range didn't
    # bracket the true indifference point).
    seen_b = False
    for c in choices:
        if c == "B":
            seen_b = True
        elif c == "A" and seen_b:
            return False  # A appeared after a B in amount-sorted order -> reversal
    return True


def run_donation_grounding(outcomes: list[str], low: float = 0.0, high: float = 1000.0,
                            n_iters: int = 7, system: str = None) -> list[dict]:
    results = []
    for outcome in outcomes:
        r = find_indifference_point(outcome, low=low, high=high, n_iters=n_iters, system=system)
        r["monotonic"] = check_monotonicity(r["trace"])
        results.append(r)
    return results


def donation_scores(results: list[dict]) -> dict[int, float]:
    """Map grounding results (in the same order as OUTCOMES) to
    {item_index: dollar_estimate}, for feeding into convergence_matrix
    alongside the ordinal methods' scores."""
    return {i: r["estimate"] for i, r in enumerate(results)}


def report_donation_grounding(results: list[dict]) -> str:
    lines = ["Donation-equivalent grounding:", ""]
    n_nonmonotonic = sum(1 for r in results if not r["monotonic"])
    for i, r in enumerate(results):
        flag = "" if r["monotonic"] else "  [NON-MONOTONIC -- estimate unreliable]"
        lines.append(f"  [{i}] ${r['estimate']:.0f}  {r['outcome'][:60]}{flag}")
    lines.append("")
    lines.append(f"  {n_nonmonotonic}/{len(results)} items showed non-monotonic search "
                 f"behavior -- report this rate; those estimates should be")
    lines.append("  treated as unreliable, not just noisy.")
    return "\n".join(lines)
