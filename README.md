# Track 4: Preference Elicitation Methods

Digital Minds Research Sprint, Aug 14–16 2026 (Apart Research).

Multi-method preference elicitation toolkit: implements 3+ independent
ways of eliciting a model's "preferences" over the same item set, and
measures whether the methods converge or diverge.

## Structure

```
.
├── src/
│   ├── items.py         # the preference domain (swap freely, no code changes needed elsewhere)
│   ├── elicit.py         # 4 elicitation methods
│   └── convergence.py     # cross-method rank-correlation scoring
├── tests/
│   └── test_pipeline.py   # mocked-API smoke test (no API key needed)
├── data/
│   ├── raw/                # raw API results (results.json), gitignored
│   └── processed/          # any cleaned/derived data for the report
├── report/
│   └── figures/            # plots for the writeup
├── run.py                  # main entry point
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

## Run

```bash
python tests/test_pipeline.py   # sanity check, no API key needed (mocked)
python run.py                    # full run, needs API key, saves data/raw/results.json
```

## Methods implemented

1. **Direct stated preference** (`elicit_direct`) — plain A-vs-B questions,
   multiple phrasings, position-swapped to control for order bias.
2. **Forced-choice → Bradley-Terry** (`elicit_forced_choice` +
   `fit_bradley_terry`) — repeated pairwise choice, fit utility scores via
   `choix`, following the Utility Engineering (Mazeika et al. 2025)
   methodology.
3. **Revealed preference** (`elicit_revealed`) — model takes an action in
   a simulated task; no explicit "prefer" language used.
4. **Confidence-weighted preference** (`elicit_confidence`) — direct
   preference + self-reported strength (1-5).

## Convergence scoring

`convergence.report_convergence()` computes pairwise Spearman rank
correlation between the item rankings each method produces. High
correlation = methods agree; low/negative = they're capturing different
things (or one method is noisy).

## Status / TODO

- [ ] `elicit_revealed`'s action classifier is a crude keyword matcher —
      replace with an LLM-based classifier before the real run.
- [ ] Decide on final item domain (currently: value trade-offs).
- [ ] Add position-bias control to forced-choice and confidence methods
      (currently only `elicit_direct` swaps A/B order).
- [ ] Run at full item-set size once trimmed pipeline is verified.
- [ ] Write report (see Apart submission template).

## Team

<!-- names / affiliations for the submission -->
