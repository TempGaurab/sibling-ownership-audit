# When ownership tests accuse the wrong model

Most open language models are fine-tuned from a handful of public bases. If you fine-tune your own model from one of those bases and then run an **ownership test** against a stranger's model built from the same base, will the test say the stranger copied you?

This repository holds the code for a preregistered study that checks exactly that, on real models.

```
                 public base model (e.g. Qwen3-0.6B)
                 /                                  \
       your model (the claimant)          someone else's model (a sibling)
                 |                         never saw your model
       a real copy of yours
       (fine-tuned, merged, quantized...)

  A good ownership test flags the copy and leaves the sibling alone.
  We measure how often published tests flag the sibling anyway.
```

## What the study does

1. **Builds siblings we know are innocent.** For two model families (Qwen3-0.6B and TinyLlama-1.1B-Chat) we trained 3 claimants and 600 siblings per family from the same base, without access to any claimant. We also collected hundreds of real sibling models from the Hugging Face Hub, frozen at a snapshot taken *before* our claimants existed, so none of them can be a copy.
2. **Builds copies we know are guilty.** Each claimant goes through 65 declared transformations: further fine-tuning, merging, quantization, pruning and distillation into a student.
3. **Runs the published tests under their own rules.** We reimplemented the tests, each labelled as a reimplementation: Nikolić et al., MPS, REEF, LLMPrint, Witness Overlap, a lineage score, task-vector cosine and a keyed fingerprint. Then we count how often each one accuses an innocent sibling.
4. **Tries a fix.** We calibrate a test against a reference set of siblings instead of unrelated models, and check on held-out groups whether that keeps false accusations near the 5% target without losing the copies.

**Short version of the findings:** most published decision rules accused siblings far more often than their stated error rates. The keyed fingerprint accused none, but it also missed many copies, including every model trained on the claimant's outputs. Calibrating against siblings worked only when the reference covered how the suspect was actually made. The paper has the numbers.

## Repository map

| Folder or file | What it is |
|---|---|
| `sim.py` | Simulations that illustrate the theory. Synthetic only, seed 2027. Runs on a laptop CPU. |
| `src/` | Everything else, from hub census to the numbers in the paper (see the pipeline below). |
| `slurm/` | The cluster jobs that ran each stage. |
| `prereg/` | Hypotheses, decision rules, codebook, deviations log, and the timestamped hash manifests with their OpenTimestamps proofs. |
| `data/` | Scores, rule outcomes, stratum codes, hub lists and aggregate results. See `data/README.md`. |

## The pipeline, step by step

| Step | What happens | Main scripts |
|---|---|---|
| 1. Census | Count derivatives of candidate bases on the Hub, from metadata only | `census.py`, `sample_hub.py` |
| 2. Hub snapshot | Download sampled siblings at pinned commits and check their weights against the declared base | `download_hub.py`, `check_weights.py`, `adapter_load.py` |
| 3. Stratum coding | Code each model card for training-data overlap with each claimant, blind to any score | `fetch_cards.py`, `build_coding_packets.py`, `merge_coding.py` |
| 4. Training | Claimants and controlled siblings on a grid of datasets, recipes and seeds | `make_grid.py`, `train_model.py` |
| 5. Copies | 65 declared copies per claimant, and keyed-fingerprint claimants | `make_copies.py`, `transform.py`, `train_student.py`, `make_keyed.py`, `train_keyed.py` |
| 6. Features | Black-box outputs, white-box weights and representations, fingerprint searches | `extract_blackbox.py`, `extract_whitebox.py`, `build_reef.py`, `llmprint.py` |
| 7. Scoring | Every method on every claimant-suspect pair | `audit.py`, `score_keyed.py`, `post_score_check.py` |
| 8. Analysis | Preregistered hypotheses H1 to H6, original decision rules, ablations | `analyze.py`, `apply_rules.py`, `rules.py`, `ablations.py` |
| 9. Post hoc checks | Fidelity of the reimplementations, extra splits, keyed copies, human check of the card coding | `fidelity_check.py`, `posthoc.py`, `make_keyed_copies.py`, `keyed_power.py`, `human_agreement.py` |
| 10. Paper | Every number, table and figure, generated from the outputs | `make_real_numbers.py`, `make_result_macros.py`, `make_si_tables.py`, `make_*_tables.py`, `make_release_data.py` |

Every script starts with a docstring that gives its inputs, outputs and usage.

## Running it

**Just the simulations** (no GPU):

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy matplotlib
.venv/bin/python sim.py
```

**The full study** needs a GPU cluster with Slurm, about 5 TB of disk (1.5 TB of hub models and about 3 TB of trained models and copies), and the Python packages listed under the system below. The jobs in `slurm/` show the order and the settings that ran. Paths inside them point at the original cluster's project folder, so set your own before submitting. Every job can be resubmitted safely, because finished work is skipped.

## Compute used

Everything ran on one Slurm cluster node, using at most 3 GPUs at a time.

| | |
|---|---|
| GPU | NVIDIA H200, 141 GB (driver 595.71) |
| CPU | 2 × AMD EPYC 9355 (32 cores each, 128 hardware threads), 1.5 TB RAM, shared with other users |
| OS | Linux 4.18 (EL8), Slurm |
| Software | Python 3.11.18, PyTorch 2.15.0 (CUDA 13.2), Transformers 5.22.2, PEFT 0.21.2, Datasets 5.1.0, NumPy 2.4.6 |
| Precision | Weights kept in float32 throughout; the LLMPrint search ran under bf16 autocast |

**H200 time by stage.** Produced by `src/gpu_hours.py` from the cluster's job records, counting only this project's jobs.

| Stage | Jobs | H200 GPU-hours |
|---|---:|---:|
| Pilot runs, timing and tests | 16 | 3.6 |
| Weight check of hub models | 3 | 1.8 |
| Training claimants and siblings | 2 | 59.8 |
| Declared copies and keyed claimants | 2 | 7.2 |
| Feature extraction (incl. control top-up) | 42 | 38.3 |
| LLMPrint fingerprint search | 6 | 81.3 |
| Scoring and analysis | 4 | 43.9 |
| Keyed copies (post hoc) | 2 | 9.0 |
| **Total** | **77** | **244.9** |

From Slurm accounting for jobs started 2026-11-06 to 2026-11-12: wall time multiplied by GPUs allocated, idle time included.

The whole study took about as much GPU time as one H200 running for ten days. The LLMPrint fingerprint search was the most expensive single step.

## Preregistration

Hypotheses, decision rules, splits and seeds were committed before any confirmatory score was computed. A manifest listing the SHA-256 of each frozen file was timestamped in Bitcoin through OpenTimestamps, six times between 6 and 9 November 2026. The manifests and their `.ots` proofs are in `prereg/timestamps/`, and `attestation_check.txt` gives the Bitcoin block of each.

To check a stamp, hash a file and compare it with its manifest line, then verify the manifest's `.ots` proof with `ots verify`. Most files in `src/`, `prereg/` and `data/hub/` are unchanged since they were stamped, which is why the cluster paths were left as they ran. A few were changed later, each change logged with its date and reason in `prereg/DEVIATIONS.md`. Their stamped versions are kept in `prereg/stamped/`, named `<file>.<first 12 hex digits of its SHA-256>`. One stamped file is not released: `data/strata/strata.json` quotes the model cards, so `data/strata/strata_final.json` holds the same codes without the quotes (see `data/README.md`).

## What is not here

- **Model weights and fingerprint keys.** The weights are large, and the keys would let anyone fake a keyed claim.
- **Third-party model card text.** The cards belong to their authors. `fetch_cards.py` refetches them at the pinned commits.
- **Per-model results for hub models under their real names.** Every hub accusation in this study is false by construction, and a named list could be read as verdicts on other people's models. Results for hub models and controls are released under pseudonyms, and the models themselves are listed by name and commit in `data/scores/third_party_models.csv`.

## Licence

MIT. See `LICENSE`.
