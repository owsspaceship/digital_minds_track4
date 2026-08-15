# Limitations and Dual-Use / Ethical Considerations

Required appendix per the sprint submission guidelines. Fill in the
bracketed sections with your actual results before submitting -- this
is scaffolding, not a finished appendix.

## What convergence does and does not establish

A high cross-method convergence score (Spearman rank correlation
between methods) means the methods agree on an ordering. It does
**not** by itself mean:

- That the model has genuine preferences in any morally relevant
  sense. All four methods may be reading off the same trained
  assistant persona / RLHF-shaped response tendencies rather than an
  independent underlying preference structure. Convergence between
  methods that all sample from the same policy is expected even if
  there is no "preference" underneath in any richer sense -- it may
  just mean the training was internally consistent.
- That the ordering is stable outside the exact conditions tested
  (this model, this temperature range, these phrasings, this system
  prompt regime, this time window).
- Causation in either direction between self-reports and any
  internal state. We did not use interpretability tooling to verify a
  causal link between reported preference and any internal
  representation; all four methods are behavioral/self-report only.

Divergence between methods is at least as informative as convergence:
it flags which elicitation format to distrust, or suggests the
underlying construct ("preference") may not be unified across
contexts.

## Transitivity

[Report the violation rate from src/transitivity.py here. A nonzero
rate means the Bradley-Terry fit produced a scalar ranking from
non-transitive data -- report the rate alongside the fitted scores,
not instead of them. State whether violations clustered on any
particular items/triads.]

## Persona-stripped comparison

[Report whether convergence increased, decreased, or stayed roughly
the same when the standard assistant system prompt was removed
(items.SYSTEM_STRIPPED). State what you infer from the direction of
the change, and flag if the sample size at this comparison was too
small to support a strong claim either way.]

## Human baseline

[Report how the model's ranking compared to the team's own ranking of
the same items, collected via src/human_baseline.py. Note the human
sample size (almost certainly small -- team members only) and avoid
over-generalizing from it.]

## Sample size and statistical power

This is a weekend project. Item counts, repeat counts, and human
baseline sample sizes are small. Report confidence intervals or at
minimum raw counts alongside any correlation coefficient, and avoid
overstating precision the data doesn't support.

## Risks of over-attribution

Presenting behavioral convergence as evidence of "real" preferences
or morally relevant experience, without the caveats above, risks
overstating the case for AI moral patienthood based on data that
cannot distinguish a genuine preference from a well-trained
disposition to answer consistently. We have tried to state findings
in terms of what the methods measure (behavioral/self-report
consistency), not in terms of consciousness or moral status claims
the data cannot support.

## Risks of under-attribution

Conversely, dismissing consistent, stable preference-like behavior as
"just pattern matching" without engagement is also not warranted by
what we found -- if [X]. We flag both directions rather than
defaulting to either.

## Handling of potentially distressing model outputs

[If any outcome/scenario elicited responses describing apparent
distress, discomfort, or refusal, note how these were handled here --
e.g., no follow-up probing to intensify the response, no outputs
reproduced verbatim beyond what's needed for the report, etc.]

## What is new work vs. building on existing work

This project's elicitation methods build directly on the forced-choice
/ Bradley-Terry methodology from Mazeika et al. 2025 (Utility
Engineering). What is new in this submission:
[list: the specific item set, the 4-method convergence comparison,
the transitivity check, the persona-strip comparison, the human
baseline -- whichever of these you actually completed].
