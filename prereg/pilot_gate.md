# Pilot gate, Nov 6, 2026

Pilot family only (`HuggingFaceTB/SmolLM2-135M`, 3 claimants, 24 siblings, 51 copies, base, 2 controls after 1 failed to load). Pilot scores are never reported as results. No decision rule, threshold or hypothesis was changed because of them.

**Decision: go**, on these conditions from `CLAUDE.md`:

- **Cluster and allocation:** met. `arcc2`, up to 3 H200s, both families training.
- **Throughput:** met. About 25 hours per family to train 603 models on one GPU. Extraction ran at about 30 s per model on a shared GPU, so about 3,200 models is about 27 GPU-hours if larger models cost the same; measure again before the array.
- **A published method end to end under its original rule:** met. Nikolic (reimplemented) and MPS (reimplemented) ran on real pilot models with both candidate pools. Thakur and Khoury's lineage score and task-vector cosine computed; their decision rules are not yet run end to end.
- **Strata:** met by merging, recorded below before scoring.

## Limits of this pilot
- Only 2 controls, so the pilot's rank comparison against controls could not reject at its threshold (minimum p is 1/3). That line of `src/pilot_check.py` tested nothing. Rates under candidate pool P1 are also uninformative with a pool that small.
- MPS ran with R = 200 and on the first 4 models per group. The rule uses R = 1000.
- Keyed fingerprint, Witness Overlap, LLMPrint and the REEF rule are not yet run.

## Defect found: quantization
Uniform min-max quantization per tensor left 4-bit copies with about 0.1% top-1 agreement with their claimant, because weight ranges reach 50 to 100 standard deviations. Such a model is not a usable copy. It is replaced before any confirmatory copy exists; see `DEVIATIONS.md`.

## Hub strata, merged before scoring
Sampled models coded `same` or `partial`: ultrachat 5, oasst1 6, gsm8k 17, before licence exclusions and the weight check. All are under the 19 needed to reject at alpha = 0.05.

- **Pooled:** all hub siblings passing the weight check.
- **Disjoint:** coded `none` for this claimant's dataset. This is the hub's low stratum, with the highest count.
- **Shared:** `same` and `partial` merged. Reported descriptively and marked unable to reject if under 19.
- **Unknown:** excluded from stratified references, kept in pooled ones, reported separately.

No merge depends on any score. High-sharing stratified claims rest on the controlled siblings.
