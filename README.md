# False accusations of sibling models by LLM ownership tests

Code for the study of how ownership-verification tests for language models behave on "siblings": independently developed models that share the claimant's public base model.

- `sim.py`: simulations (synthetic only; seed 2027).
- `src/`: hub census and sampling, model training and declared copies, feature extraction, reimplementations of the audited methods (each labelled as a reimplementation), the keyed fingerprint, analysis under the preregistered rules, post hoc analyses, and the generators of every number, table and figure in the paper.
- `slurm/`: the cluster jobs that ran them.

Source files are kept byte-identical to the versions whose hashes were timestamped before confirmatory scoring, so cluster paths are left as they ran. Data, model weights, fingerprint keys and third-party model card text are not included here.

Licence: MIT.
