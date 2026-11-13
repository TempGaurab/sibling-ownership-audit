# Data

Scores, stratum codes and aggregate results of the real-model study. No model weights, fingerprint keys or third-party model card text are included.

## Naming

- **Our own models** keep their real IDs. Claimants are `c0_ultrachat`, `c1_oasst1` and `c2_gsm8k`. Controlled siblings are `<claimant>_<stratum>_<n>`, copies are `<claimant>_<transformation>_<n>`, and keyed claimants are `keyed_<claimant>`.
- **Third-party models** (Hugging Face Hub siblings and independent controls) appear in the per-model result files only as pseudonyms `hf-NNNN`, and their publishers as `pub-NNNN`. A model has the same pseudonym in both families. Rows are sorted by pseudonym, so their order says nothing about names. Every hub accusation in this study is false by construction, since our claimants were trained after the hub snapshot, and a named list could be read as verdicts on other people's models. The mapping was drawn at random (not from a seed) and is available from the corresponding author for reproduction.
- `scores/third_party_models.csv` lists every third-party model by its real repository name and pinned commit, so the sample can be rebuilt.

The pseudonyms stop the release from reading as a list of named verdicts. They are not a guarantee against deliberate re-identification, because upload month, stratum and publisher grouping are kept for the analyses.

## Files

| Path | Contents |
|---|---|
| `scores/scores_F.csv` | One row per claimant, suspect and score, for family F (`qwen3`, `tinyllama`): `value`, suspect `kind` (`sib`, `hub`, `control`, `copy`), `stratum`, `dataset`, `recipe`, `publisher`, upload `month` |
| `scores/rules_F.jsonl` | Outcomes of the original decision rules (Nikolić et al. P1/P2, MPS set membership and Non-Infringement Score) per claimant and suspect |
| `scores/llmprint_F.jsonl` | LLMPrint fingerprint bits of each model against each claimant's fingerprint prompts |
| `scores/keyed_F.jsonl` | Keyed-fingerprint hit rates of each model against each claimant's key |
| `scores/own_models_F.csv` | Our models: kind, base, base commit, adapter or full weights |
| `scores/third_party_models.csv` | Hub siblings and controls: repository, pinned commit, declared base, format, families, whether scored, final stratum codes |
| `scores/SHA256SUMS` | Hashes of the files above |
| `strata/strata_final.json` | Final stratum codes of the 1,600 sampled hub models, by repository name |
| `strata/pass1/`, `strata/pass2/` | The two blind LLM coding passes, keyed by card group (`groups.json` maps groups to repositories) |
| `strata/agreement.txt`, `strata/audit.txt` | Repeatability of the two passes and the keyword audit |
| `strata/human_sample.json`, `human_codes.csv`, `human_agreement.json` | Post hoc check: stratified sample, one author's codes, agreement with the LLM codes |
| `strata/CODING_INSTRUCTIONS.md` | Instructions given to each coding pass, with `prereg/codebook.md` |
| `hub/download_list.json`, `hub/topup_list.json` | The sampled hub models and control reserves at pinned commits, as timestamped |
| `hub/weight_check_v2.jsonl` | Check of each downloaded model's weights against its declared base, as timestamped |
| `census/summary.json`, `census/sample_*.json` | Hub census of candidate bases and the metadata of each sampled derivative (Nov 6, 2026 snapshot) |
| `analysis/` | Aggregate outputs: `analysis.json` (H1 to H6), `rules.json`, `ablations.json`, `posthoc.json`, `fidelity.json`, logs, and `provenance.txt` with the hashes of the cluster inputs |

## What was removed

- The `evidence` field of every stratum code (a quote of up to 15 words from the model card) and the free-text notes in `human_codes.csv`, because card text belongs to its authors. `src/fetch_cards.py` refetches the cards at the pinned commits. The timestamped `data/strata/strata.json` contains these quotes, so it is not included here; its hash is in the manifests in `prereg/timestamps/`, and the file is available from the corresponding author. `strata/strata_final.json` is the same file without the quotes.
- Paths on the original cluster. `provenance.txt` and `hub/weight_check_v2.jsonl` keep them, as they were hashed.

## Rebuilding

`src/make_release_data.py` produced `scores/` from the cluster feature files. The released files equal the originals row for row once the pseudonyms are mapped back. The aggregate files in `analysis/` were computed from the originals by `src/analyze.py`, `src/apply_rules.py`, `src/ablations.py`, `src/posthoc.py` and `src/fidelity_check.py`. Point estimates do not depend on names. A rerun on the released files has not been checked digit for digit: row order differs from the originals, which can change resampling draws in bootstrap intervals.

Keyed-fingerprint scores of the post hoc keyed copies will be added when that run finishes.
