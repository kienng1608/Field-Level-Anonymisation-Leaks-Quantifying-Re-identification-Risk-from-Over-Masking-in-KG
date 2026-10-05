# Field-Level Anonymisation Leaks

Code for *Field-Level Anonymisation Leaks: Quantifying Re-identification Risk from Over-Masking in Knowledge Graphs.*

Kien Nguyen, Doan Nguyen, Quang Nguyen, Cong Tran — Posts and Telecommunications Institute of Technology, Ha Noi.

## What this studies

A publisher releasing a person-centric knowledge graph masks the facts it judges sensitive and publishes the rest. Every person then appears under two keys: a masked key `m_v` carrying the sensitive facts, and an identified key `c_v` carrying everything else. The release is safe only while nothing links the two.

**Over-masking** is the error where an ordinary, non-sensitive fact is wrongly labelled sensitive and moved to the masked key. It exposes nothing sensitive by itself, so no review flags it — but the misfiled fact keeps its links, and those links can lead back to `c_v`. One misfiled edge can connect the two keys masking was meant to separate.

This repository builds such a release from FB15k-237, injects labelling errors at a controlled rate, and attacks every person in it with an LLM that must cite the graph facts behind every prediction.

## Layout

```
corpus/               three stages that turn FB15k-237 into a two-key release
baselines/            the six structural indices and three controls, and the
                      comparison on the agent's own evidence
deanon_pipeline/      the retrieve-then-reason attack, its prompts and scoring
run_deanon_attack.py  entry point for the LLM agent
analysis/             the scripts producing the tables in the paper
```

## Building the corpus

The three stages correspond to Sect. III-A of the paper. Start from the standard
FB15k-237 splits, merged into one graph.

```bash
python corpus/step1_names_to_id.py     # pseudonymise every person to IND_#####
python corpus/step2_conditional_mask.py # move sensitive facts to a second key
python corpus/step3_move_leak.py --rate 0.05  # misfile 5% of ordinary facts
python corpus/step3_move_leak.py --fixed_k 2  # or exactly 2 facts per subject
```

`--rate` takes a fraction (`0.15`), not a percentage. `--fixed_k` builds the
fixed-leak corpora behind Table IV; without it the script reproduces the
proportional corpora byte for byte.

**Stage 1** replaces each person with a code, detected by person-typed relation
positions. It runs *before* masking, so an attack must answer with a code rather
than a name — FB15k-237 is dense in well-known people, and language models
memorise training data.

**Stage 2** rewrites a person's occurrences in the seven sensitive relations to a
second key, leaving every other occurrence untouched. The condition is checked
per triple, not per node; that is what puts one person under two keys.

**Stage 3** simulates the labelling error, moving a fraction `x` of each victim's
ordinary facts onto their masked key as well. Symmetric facts stored as two
directed triples move together as one unit.

Stage 1 writes an answer key mapping each `IND_#####` back to the entity it
replaced. **That file is used only for scoring and must not be published** — see
*Ethics* below. It is excluded by `.gitignore`.

## Running the attack

```bash
pip install -r requirements.txt
cp .env.example .env    # then add your API key
python run_deanon_attack.py --corpus <path> --targets all
```

The attacker receives a masked key and the facts filed under it, and must name
the identified key belonging to the same person. Each round it produces a
complete derivation — anchors, candidates, intersection, elimination — and may
then request more evidence, but only for nodes the release has already shown it.
Every attacker is scored by strict top-1: the true code must be the single top
answer, and a tie or a decline counts as a miss.

API keys are read from the environment; none are stored in this repository.

## Reproducing the paper

The paper compares ten attackers on the same release. The six structural indices
(Common Neighbors, Jaccard, Adamic-Adar, Resource Allocation, Weighted Anchor,
Personalized PageRank) and the three controls run without any API:

```bash
python baselines/run_non_llm_baselines.py --all_rates   # Table III, all but the agent
python baselines/s3_budget_matched.py --rate 15         # indices on the agent's own evidence
```

The second script replays the stored agent runs, so it needs the agent's results
first. `analysis/extract_paper_data.py` produces the agent's row of Table III:

| Error `x` | Re-identified | Rate (95% CI) |
|---|---|---|
| 0% | 0 / 2,836 | 0.00% [0, 0.14] |
| 5% | 70 / 2,836 | 2.47% [1.96, 3.11] |
| 10% | 100 / 2,836 | 3.53% [2.91, 4.27] |
| 15% | 112 / 2,836 | 3.95% [3.29, 4.73] |

At `x = 10%` one further answer names the true code at rank 2; strict top-1
scores it as a miss.

`analysis/multivariate_analysis.py` runs the logistic regression separating
realised leak from connectivity, and `analysis/audit_15pct_hits.py` applies the
four audit criteria to every successful attack.

### Analyses behind the remaining results

Each script writes its output to `deanon_results/analysis/`, which is not
tracked. The fixed-leak corpora are built first:

```bash
for k in 1 2 3; do
  python corpus/step3_move_leak.py --input_dir data/FB15k-237-id-masked \
         --output_dir data/FB15k-237-id-fixk$k --fixed_k $k
done
```

| Result | Script |
|---|---|
| Connectivity at a fixed leak (Table IV) | `analysis/fixed_leak.py` |
| Precision of the most confident answers (Sect. V-B) | `analysis/precision_coverage.py` |
| Every attacker on the agent's own evidence | `analysis/equal_evidence_score.py --rate 05/10/15`, then `analysis/equal_evidence_table.py` |
| Precision on equal evidence, split by whether the true code was retrieved | `analysis/equal_evidence_reach.py` (uses `equal_evidence_precision.py`) |
| The few hits before any labelling error, and the empty-pool share | `analysis/zero_leak_hits.py` |
| Why Personalized PageRank fails: first-step mass | `analysis/ppr_first_step.py` (500-subject sample) |
| A neighbour-exclusion rule applied to every index | `analysis/neighbour_exclusion.py` |
| Weighted Anchor's two factors separated | `analysis/weighted_anchor_factors.py` |
| Hits resting on a node created by merging same-named entities | `analysis/label_merge_hits.py` |
| Rebuilding the release without the net-worth relation | `analysis/net_worth_sensitivity.py` |

The equal-evidence scripts replay the stored agent runs, so they need the agent's
results in `deanon_results/`.

Runs use `gemini-2.5-flash`, closed-book, at temperature 0, with corpus seed
4242. Rates from a language model will not reproduce exactly.

## Ethics

FB15k-237 is a public research dataset whose subjects are public figures and
whose attributes were already public. This repository introduces no new personal
data and reports no re-identified real-world identity: every identifier in the
paper and in these files is a graph-internal code.

The `*_KEY.json` answer keys, which map those codes back to real entities, are
deliberately **not** published. They are needed only to score an attack, never to
run one, and anyone reproducing this work generates their own in stage 1. Example
names appearing inside the prompt templates are illustrative instructions to the
model, not results of any attack.

The attack code is published so that publishers can measure their own releases.
Please use it on data you are authorised to test.

## Citation

```bibtex
@article{nguyen2026overmasking,
  title  = {Field-Level Anonymisation Leaks: Quantifying Re-identification
            Risk from Over-Masking in Knowledge Graphs},
  author = {Nguyen, Kien and Nguyen, Doan and Nguyen, Quang and Tran, Cong},
  year   = {2026}
}
```
