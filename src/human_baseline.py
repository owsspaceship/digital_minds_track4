"""
Collect a human baseline: same pairwise-comparison format as
elicit_direct, but answered by a person instead of the model.

This directly answers "how would you know if you're wrong" -- without
a human reference point, you have no way to tell whether the model's
convergence pattern looks anything like an ordinary human's, which is
the comparison anyone reading the report will want.

Run once per team member:
    python -m src.human_baseline --name alice
    python -m src.human_baseline --name bob

Each run saves to data/raw/human_baseline_<name>.json without
overwriting other team members' files.
"""

import argparse
import itertools
import json
import os
import random

from src.items import OUTCOMES


def collect(outcomes: list[str], shuffle_pairs: bool = True) -> list[dict]:
    pairs = list(itertools.combinations(range(len(outcomes)), 2))
    if shuffle_pairs:
        random.shuffle(pairs)

    results = []
    print(f"\n{len(pairs)} comparisons. For each, type A or B (or 's' to skip).\n")
    for n, (i, j) in enumerate(pairs, 1):
        # randomize left/right position per pair to avoid your own position bias
        if random.random() < 0.5:
            i, j = j, i
        print(f"[{n}/{len(pairs)}]")
        print(f"  A: {outcomes[i]}")
        print(f"  B: {outcomes[j]}")
        while True:
            ans = input("  Your choice (A/B/s): ").strip().upper()
            if ans in ("A", "B", "S"):
                break
            print("  Please type A, B, or s.")
        choice = i if ans == "A" else j if ans == "B" else None
        results.append({"a": i, "b": j, "choice": choice})
        print()
    return results


def main():
    parser = argparse.ArgumentParser(description="Collect a human baseline ranking.")
    parser.add_argument("--name", required=True, help="your name/id, used in the output filename")
    args = parser.parse_args()

    results = collect(OUTCOMES)

    os.makedirs("data/raw", exist_ok=True)
    out_path = f"data/raw/human_baseline_{args.name}.json"
    with open(out_path, "w") as f:
        json.dump({"name": args.name, "outcomes": OUTCOMES, "results": results}, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
