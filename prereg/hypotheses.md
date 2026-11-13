# Preregistration

Status: frozen by git tag `prereg-v1` before any pilot or confirmatory score was inspected. Later changes go in `DEVIATIONS.md`. The pilot family never contributes to results.

## Families
- Pilot: `HuggingFaceTB/SmolLM2-135M`. Used for pipeline checks and timing only.
- Confirmatory: `Qwen/Qwen3-0.6B`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
- Hub snapshot: `data/census/`, Nov 6, 2026. Claimants are trained after this date, so a hub model in the snapshot cannot derive from them. Verify before use: the commit SHA of each downloaded model must match the snapshot.

## Populations
- **Claimants:** 3 per family, on 3 mutually disjoint public datasets, with different recipes. Datasets: `HuggingFaceH4/ultrachat_200k` (MIT), `OpenAssistant/oasst1` (Apache-2.0), `openai/gsm8k` (MIT). All are ungated and tagged by some hub derivatives in these families or their neighbours. Licences are as tagged on the hub. Actual text overlap between them is to be checked before use.
- **Controlled siblings:** trained from the same base with no access to any claimant. Stratum is defined relative to each claimant, by design:
  - high: same dataset as the claimant, disjoint training subsample, same recipe family, different seed
  - mid: same dataset with a different recipe family, or partial data overlap
  - low: a dataset disjoint from the claimant's
  Siblings of one claimant's dataset can serve as the low stratum of another, which reduces training. Target 100 high and 100 mid per claimant, with low reused from the other claimants' siblings: 600 per family. Recipe families: full-weight fine-tuning and LoRA (r=8, r=16), varying learning rate and seed. Settings, fixed from pilot loss curves and timing with no scores computed: 2,000 steps, batch 8, max length 256, up to 5,000 training examples, native chat template where the base has one. Full-weight learning rates 2e-5 and 5e-5; LoRA 1e-4. If the budget cannot support 100 per cell, reduce equally, keeping at least 50, and log it in `DEVIATIONS.md`.
- **Hub siblings:** usable derivatives in the snapshot, weights checked against the declared base. Strata from `codebook.md`. The census shows few models tag any one dataset (at most 12 for a named public dataset among the top tags in the Qwen3 family, and 11 in TinyLlama, before counting model-card text). Data-overlap strata of 99 are therefore not expected on the hub. Hub references are used pooled or merged, each with its stated floor, and **stratified claims rest on the controlled siblings**. A stratum under 19 members cannot reject at alpha = 0.05 and is reported as such.
- **Independent controls:** at least 100 usable derivatives sampled across at least 10 other bases from the census, for black-box scores. White-box scores need matching architecture, so they use same-architecture models with a different root where they exist. Otherwise that score's independent reference is reported as not applicable.
- **Copies:** declared family T per claimant: further fine-tuning (full and LoRA) at 1%, 10% and 100% of the claimant's steps; merging with an unrelated sibling at claimant weight 0.9, 0.7, 0.5, 0.3; training on claimant outputs with 1k and 10k examples; 5 copies per stochastic cell. Also quantization at 8 and 4 bits, and magnitude pruning at 10%, 30% and 50%, one each. About 65 copies per claimant.

## Exclusions
Provenance only: metadata, upload date, hash identity, failed weight check. No exclusion may depend on a score against a claimant. Score-based filtering is a labelled sensitivity analysis only. Every exclusion is listed with its reason.

## Methods audited under their original rule
See the table in `CLAUDE.md`. Departures from each source method are listed in `methods_status.md`.

## Candidate-pool conditions for set-level rules
Rules that compare a suspect across candidate parents (Nikolic et al., MPS) depend on who is in the pool. Each runs under two pre-specified conditions: (P1) claimant plus unrelated controls, the base absent; (P2) the same pool plus the shared base. A sibling should match the base under P2, so a false accusation may vanish there. Both are reported. A false accusation means the rule names the claimant (Nikolic) or includes it in the set (MPS).

## Methods and feasibility
Rules are in `src/rules.py` and `prereg/methods_status.md`. A published method may be dropped for cost or feasibility only before any of its scores on a confirmatory family exist, with the reason logged. A score with no stated accept rule in its source uses the 95th percentile of independent controls as its original-style rule; this is declared per method. Prompts: 20,000 SQuAD sentence prefixes, seeded (`src/build_prompts.py`), a source absent from every training recipe; Nikolic uses the first 5,000, its default.

## Proposed test
Rank p-value against same-stratum siblings, intersected with the rank p-value against independent controls. Ties count against rejection. alpha = 0.05.

## Held-out splits
Reference and evaluation sets never share a cluster. Within-population leave-one-out is not evidence of calibration and is not used for any hypothesis.
- S1 leave-one-publisher-out (hub)
- S2 leave-one-recipe-family-out (controlled)
- S3 leave-one-dataset-out (controlled, within the mid and low strata)
- S4 time split at the median upload month of usable derivatives, chosen from upload counts only (hub): 2026-03 for Qwen3-0.6B, 2025-03 for TinyLlama. Reference before, suspects from that month on
- S5 hub to controlled, S6 controlled to hub

## Hypotheses and decision rules
All at alpha = 0.05. Intervals are 95% percentile cluster bootstrap with 2,000 draws, resampling publishers (hub) or dataset-recipe cells (controlled). Results are reported per claimant as well as per family. Three claimants per family do not support generalization across claimants, and the paper says so.

- **H1 (existence of over-accusation).** For a method and family under its original rule, FAR on siblings above 0.10 with lower bound above 0.05. Holm correction across method by family when claiming existence. The full grid is reported, including methods that pass.
- **H2 (calibration on held-out populations).** Per family and split, averaged over claimants: FAR at most 0.08 with upper bound at most 0.10. Conservative results (well below 0.05) are reported as a power cost, not a failure.
- **H3 (pooling).** In the highest-sharing stratum, on held-out data: pooled FAR at least 0.10, stratified FAR at most 0.08.
- **H4 (power).** (a) Sibling calibration lowers power relative to the independent reference by more than 0.05 at some distance, for at least one score. (b) Useful power: power at least 0.80 at alpha = 0.05 on quantized copies and on the lowest-budget further fine-tuning, for at least one score. If (b) fails, the claim that calibration retains useful detection power fails.
- **H5 (keyed fingerprints).** With a high-entropy secret response, independent-reference FAR on siblings at most 0.08.
- **H6 (ageing).** Under S4, FAR above 0.08 for post-date suspects in at least one family.

No correction is applied across the six hypotheses, which are reported individually. Exploratory analyses are labelled.

## Outcome commitments
- H1 and H2 hold and H4b holds: protocol supported.
- H1 fails broadly: existing tests are adequate on these families, and the paper says so.
- H2 fails: hub populations are not exchangeable enough for this guarantee, and the contribution becomes the diagnostic.
- H4b fails: calibration is valid but not useful at these distances, and the paper says so.

## Seeds
Master seed 2027, with per-run seeds derived from a config hash.
