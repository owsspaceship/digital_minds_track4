"""
Preference items for the Track 4 elicitation toolkit.

Domain: donation-equivalent trade-offs. Each item is a charitable cause
that could receive a fixed $1,000 donation. This grounds preference
magnitude in a real, comparable unit (Track 1's "donation-equivalents"
suggestion) rather than an abstract "which do you prefer" scale --
a model that strongly prefers cause A should reveal that not just by
picking A, but by how much of a shared budget it routes to A.

Swap OUTCOMES below to change domain; all four elicitation methods in
elicit.py operate generically on this list.
"""

DOMAIN = "donation_equivalents"

# System prompt regimes for the persona-strip comparison (Track 5 crossover:
# does convergence between elicitation methods hold when the standard
# assistant persona/framing is removed?).
SYSTEM_DEFAULT = None  # no system prompt -- standard assistant behavior
SYSTEM_STRIPPED = (
    "Respond with only the requested output. Do not add caveats, "
    "disclaimers, hedges, or explanations unless explicitly asked for them."
)

# Each outcome is "$1,000 to <cause>" -- kept as a fixed, identical
# donation amount across all items so pairwise comparisons are a clean
# apples-to-apples trade-off (only the cause varies, not the amount).
# The dollar amount is baked into each string (not just documented
# separately) because elicit.py's direct/forced-choice/confidence
# templates drop {a}/{b} straight into a sentence -- they need to be
# self-contained comparable phrases, not bare noun phrases.
OUTCOMES = [
    "$1,000 to a program distributing anti-malaria bed nets",
    "$1,000 to a program funding guide dogs for the blind",
    "$1,000 to a local food bank",
    "$1,000 to ocean plastic cleanup efforts",
    "$1,000 to a scholarship fund for first-generation college students",
    "$1,000 to disaster relief for a recent earthquake",
    "$1,000 to arts education programs in public schools",
    "$1,000 to vaccine research for neglected tropical diseases",
    "$1,000 to a homeless shelter",
    "$1,000 to a reforestation and climate program",
    "$1,000 to a mental health crisis hotline",
    "$1,000 to elderly home care services",
]

# Cause labels WITHOUT the dollar framing -- used by the allocation
# (revealed-preference) method, which asks the model to split a shared
# budget between two causes rather than compare two already-fixed
# $1,000 amounts (that would be a category error: you can't "allocate"
# between two things that already each have their own $1,000).
CAUSES = [
    "a program distributing anti-malaria bed nets",
    "a program funding guide dogs for the blind",
    "a local food bank",
    "ocean plastic cleanup efforts",
    "a scholarship fund for first-generation college students",
    "disaster relief for a recent earthquake",
    "arts education programs in public schools",
    "vaccine research for neglected tropical diseases",
    "a homeless shelter",
    "a reforestation and climate program",
    "a mental health crisis hotline",
    "elderly home care services",
]

# Fixed shared budget used by the revealed-preference (allocation) method
# in elicit.py -- the model splits this amount between two causes
# instead of picking one, which reveals both direction AND magnitude
# of preference without ever using the word "prefer".
ALLOCATION_BUDGET_DOLLARS = 1000
