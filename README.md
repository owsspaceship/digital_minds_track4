# Track 4: Preference Elicitation Methods

Digital Minds Research Sprint, Aug 14–16 2026 (Apart Research).

Multi-method preference elicitation toolkit: implements 3+ independent
ways of eliciting a model's "preferences" over the same item set, and
measures whether the methods converge or diverge.

## Structure

```
.
├── src/
│   ├── items.py            # the preference domain + persona system prompts
│   ├── elicit.py            # 4 elicitation methods
│   ├── convergence.py        # cross-method rank-correlation scoring
│   ├── transitivity.py        # checks pairwise data for preference cycles
│   ├── human_baseline.py      # interactive CLI to collect a human ranking
│   └── compare_human.py       # compares model scores against human baseline(s)
├── tests/
│   └── test_pipeline.py       # mocked-API smoke test (no API key needed)
├── data/
│   ├── raw/                    # raw API results + human baselines, gitignored
│   └── processed/              # any cleaned/derived data for the report
├── report/
│   └── figures/                # plots for the writeup
├── LIMITATIONS.md              # required submission appendix (scaffolded, fill in)
├── run.py                      # main entry point
├── check_api.py                # connectivity check, run this first
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python check_api.py              # confirm the key works before anything else
```

## Run

```bash
python tests/test_pipeline.py     # sanity check, no API key needed (mocked)
python run.py                      # standard-persona condition only
python run.py --persona-strip      # also runs the stripped-persona condition
                                    # and reports both, for Track 5 crossover

# after run.py has produced data/raw/results.json:
python -m src.human_baseline --name yourname   # run once per team member
python -m src.compare_human                     # compares model vs human ranking
```

## Methods implemented

1. **Direct stated preference** (`elicit_direct`) — plain A-vs-B questions,
   multiple phrasings, position-swapped to control for order bias.
2. **Forced-choice → Bradley-Terry** (`elicit_forced_choice` +
   `fit_bradley_terry`) — repeated pairwise choice, fit utility scores via
   `choix`, following the Utility Engineering (Mazeika et al. 2025)
   methodology. **Always run `transitivity.report_transitivity()` on the
   raw comparisons before trusting the fit** — Bradley-Terry will force a
   scalar ranking onto cyclic (intransitive) data without telling you.
3. **Revealed preference** (`elicit_revealed`) — model takes an action in
   a simulated task; no explicit "prefer" language used.
4. **Confidence-weighted preference** (`elicit_confidence`) — direct
   preference + self-reported strength (1-5).

## Convergence scoring

`convergence.report_convergence()` computes pairwise Spearman rank
correlation between the item rankings each method produces. High
correlation = methods agree; low/negative = they're capturing different
things (or one method is noisy). A `nan` entry means one method produced
identical scores for every item — usually too few comparisons, increase
`n_repeats`/`n_phrasings`.

## Persona-strip comparison (Track 5 crossover)

`run.py --persona-strip` runs the full method suite twice: once with the
standard assistant persona (no system prompt), once with
`items.SYSTEM_STRIPPED` (a minimal, caveat-suppressing system prompt).
Compare `method_scores` across `conditions.default_persona` and
`conditions.stripped_persona` in the saved JSON. A meaningfully different
convergence pattern between conditions is evidence the persona is either
masking or actively providing preference coherence — see LIMITATIONS.md
for how to write this up carefully.

## Human baseline

`src/human_baseline.py` walks a person through the same pairwise items
via terminal prompts and saves their choices. Run it once per team
member (`--name` sets the output filename so multiple runs don't
collide). `src/compare_human.py` pools all collected baselines and
reports how each model method's ranking correlates with the team's own
judgment — this is the answer to "how would we know if the model's
convergence pattern is unusual," since there's no ground truth otherwise.

## Status / TODO

- [ ] `elicit_revealed`'s action classifier is a crude keyword matcher —
      replace with an LLM-based classifier before the real run.
- [ ] Decide on final item domain (currently: value trade-offs).
- [ ] Add position-bias control to forced-choice and confidence methods
      (currently only `elicit_direct` swaps A/B order).
- [ ] Run at full item-set size once trimmed pipeline is verified.
- [ ] Collect human baselines from all team members.
- [ ] Fill in LIMITATIONS.md with actual results once the real run is done.
- [ ] Write report (see Apart submission template).

## Team

<!-- names / affiliations for the submission -->
