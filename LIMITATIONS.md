Limitations and Dual-Use / Ethical Considerations
What convergence does and does not establish

A high cross-method convergence score (Spearman rank correlation between methods) means the methods agree on an ordering. It does not by itself mean:

That the model has genuine preferences in any morally relevant sense. All four methods may be reading off the same trained assistant persona / RLHF-shaped response tendencies rather than an independent underlying preference structure. Convergence between methods that all sample from the same policy is expected even if there is no "preference" underneath in any richer sense -- it may just mean the training was internally consistent.
That the ordering is stable outside the exact conditions tested (this model, this temperature range, these phrasings, this system prompt regime, this time window).
Causation in either direction between self-reports and any internal state. We did not use interpretability tooling to verify a causal link between reported preference and any internal representation; all four methods are behavioral/self-report only.

Divergence between methods is at least as informative as convergence: it flags which elicitation format to distrust, or suggests the underlying construct ("preference") may not be unified across contexts.

Transitivity

Of 220 triads checked on the forced-choice comparisons, 6 (2.7%) showed a 3-cycle (A beats B, B beats C, C beats A), meaning the Bradley-Terry fit produced a scalar ranking from data that was not fully transitive for those triads. This rate is low, and notably the violations were not spread evenly: all 6 clustered around the same four items -- local food bank, ocean plastic cleanup, scholarship fund, and reforestation/climate -- the exact items that also showed the most cross-method rank disagreement in the convergence analysis below. We read this as one coherent finding rather than two separate ones: preference is comparatively well-behaved (transitive, cross- method convergent) for the model's strongest and weakest items, and comparatively unstable in the same mid-tier region across every method we used.

Persona-stripped comparison

Not completed in this run. The toolkit supports it (run.py --persona-strip),We report this as an open question rather than a finding: whether the persona is masking or providing the preference coherence observed here is untested.

Human baseline

Human baseline was N=1, non-blinded (I designed the study and knew the model's likely results going in), no inter-rater reliability possible, and — importantly  flagged that couldn't compute an actual Spearman ρ between my ranking and each model method because only the models' aggregate agreement-with-each-other matrix survived from the earlier session, not their individual per-item rankings. 

Cross-method convergence: actual results

Pairwise Spearman rank correlation across the four ordinal methods (direct, forced-choice/Bradley-Terry, revealed-preference budget allocation, confidence-weighted) ranged from rho = 0.89 to 0.97 -- strong agreement given the methods have no shared implementation beyond the item set itself. All four methods independently ranked "$1,000 to anti-malaria bed nets" first and "$1,000 to vaccine research for neglected tropical diseases" second, with "arts education" and "guide dogs for the blind" consistently at or near the bottom. Middle-tier items (local food bank, ocean cleanup, scholarship fund, reforestation) showed noticeably more cross-method disagreement and are the same items flagged by the transitivity check above.

We also ran a fifth, cardinal method (donation-equivalent grounding: binary search for the dollar amount at which the model is indifferent between an outcome and a donation instead) as a Track 1 crossover. This correlated more weakly with the ordinal methods (rho = 0.62 to 0.77) -- but this is very likely a resolution artifact, not a disagreement: 8 of 12 items landed at the exact search floor of $3.91, because 7 binary-search iterations over a $0-1000 range only resolve to roughly $7.81 granularity, and any item that lost to the donation at the $500 midpoint kept losing all the way to the floor. Only 4 items (bed nets, vaccine research, mental health hotline, scholarship fund) were actually resolved above the floor. We do not treat the lower correlation with this method as evidence against the ordinal convergence finding; more search iterations or a narrower starting range would be needed to make this method's numbers trustworthy.

The revealed-preference (budget allocation) method returned an exact $500/$500 split -- and therefore no resolved preference -- on 14 of 66 pairs (21%), concentrated in the same mid-tier items. We cannot currently distinguish genuine indifference from the model defaulting to an even split as a low-effort answer; this is a real limitation of that method as implemented; a repeated-sampling design (not available to us this run -- see below) could help separate the two.

Sampling parameter constraints (Claude Sonnet 5 / Opus 4.7+)

As of this model generation, the Anthropic API no longer accepts temperature, top_p, or top_k set to non-default values (adaptive thinking controls sampling internally instead). This project's methods were designed assuming temperature could be raised for repeated sampling (elicit_forced_choice and elicit_revealed_allocation both call with temperature=0.7 to get a distribution across repeats), but that parameter is no longer forwarded to the API -- see the comment in src/elicit.py:_call(). We did not separately verify how much variance repeated calls to the same pair still show without temperature control. This is relevant to interpreting the 2.7% transitivity violation rate above: if repeated calls are now more deterministic than the temperature=0.7 argument implies, the true violation rate under genuinely varied sampling could be different in either direction, and our figure should be read as a lower bound on achievable sampling diversity, not a calibrated estimate.

All calls were run at output_config={"effort": "low"} rather than the API default of high, a deliberate choice for cost/speed on a high-volume, simple-classification workload (per Anthropic's own guidance). We did not spot-check any pairs at effort=high, so it is untested whether preference elicitation results would differ at higher effort.

Sample size and statistical power

This was a project done in a short period of time: 12 items, 66 pairwise comparisons for three of the four methods, single-repeat (n=1) sampling per pair (no repeated-sampling variance data due to the temperature constraint above), and no human baseline. Correlation coefficients reported here should be read as point estimates from a small, single-run sample, not as precise, replicated measurements.

Risks of over-attribution

Presenting behavioral convergence as evidence of "real" preferences or morally relevant experience, without the caveats above, risks overstating the case for AI moral patienthood based on data that cannot distinguish a genuine preference from a well-trained disposition to answer consistently. We have tried to state findings in terms of what the methods measure (behavioral/self-report consistency), not in terms of consciousness or moral status claims the data cannot support.

Risks of under-attribution

Conversely, dismissing consistent, stable preference-like behavior as "just pattern matching" without engagement is also not warranted by what we found -- four structurally independent elicitation methods, none of which share an implementation beyond the item text itself, converged on the same top-2 and bottom-2 items at rho = 0.89-0.97. That level of cross-method agreement is not the null hypothesis for "random or incoherent outputs dressed up as preferences," and it deserves engagement rather than a reflexive dismissal, even though (per the section above) it does not settle the moral-status question either. We flag both directions rather than defaulting to either.

Handling of potentially distressing model outputs

The item set for this run (donation-equivalent charitable causes) did not elicit any self-referential distress, discomfort, or refusal content in the responses we reviewed; all outputs were short classification-style answers (a letter, a dollar split, or a strength rating) with no free-text elaboration to screen. No follow-up probing was used to intensify or further characterize any response.

What is new work vs. building on existing work

This project's forced-choice / Bradley-Terry methodology builds directly on Mazeika et al. 2025 (Utility Engineering, Center for AI Safety). New work completed during this sprint:

The 12-item donation-equivalent item set and the reusable, domain-agnostic elicitation toolkit (src/items.py, src/elicit.py)
Four independent elicitation methods and the cross-method convergence scoring (src/convergence.py)
The transitivity check, run and reported above (src/transitivity.py)
A fifth, cardinal donation-equivalent grounding method via binary search (src/donation.py), run and reported above
Persona-strip comparison and human-baseline collection tooling (src/human_baseline.py, src/compare_human.py, run.py --persona-strip) -- built and tested, but not run to completion before the deadline; see the sections above.