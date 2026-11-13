# Audit methods: code and rule status (Nov 6, 2026)

| Method | Code | Licence | Plan |
|---|---|---|---|
| Nikolic et al. | github.com/ivicanikolicsg/model_provenance_testing | none | Reimplement rule from paper and source reading: 5,000 prompts, top-1 next token, Holm step-down against the other candidates, alpha 0.05. Do not vendor. Own prompt set (their prompts are unlicensed) |
| MPS | anonymous link expired on Nov 6 | | Reimplement Token-MPS from Algorithm 1. Own prompts (theirs came from closed LLMs). Declared departure |
| Witness Overlap | anonymous link expired on Nov 6 | | Reimplement from paper. Threshold from leave-one-family-out logistic regression, using the other families |
| Thakur and Khoury | not found yet | | Reimplement from paper: RELATED if lineage score exceeds the maximum of independent nulls. It answers shared ancestry, so siblings are expected positives, and the paper warns against using it as a legal finding. Report as "what happens if used for ownership", not as a flaw in the method |
| REEF | github.com/AI45Lab/REEF | none | Reimplement representation CKA with a threshold rule per the paper. Do not vendor |
| LLMPrint | github.com/hifi-hyp/ACL-LLMPrint | none | Reimplement GCG suffix search and mu + 1.64 sigma rule. Measure cost on the pilot first |
| Keyed fingerprint | not chosen | | Implement a simple high-entropy key-response implant, marked as ours, or adopt a licensed method |
| Task-vector cosine | ours | | Trivial |

All reimplementations are labelled as such in the paper. Where feasible, each is sanity-checked against a result from its source paper.

"No licence" means no stated right to reuse. Do not vendor or redistribute that code.
