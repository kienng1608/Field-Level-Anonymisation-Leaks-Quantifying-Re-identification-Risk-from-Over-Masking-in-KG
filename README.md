# Field-Level Anonymisation Leaks: Quantifying Re-identification Risk from Over-Masking in Knowledge Graphs

This repository contains the implementation and research associated with measuring
how much re-identification risk a field-level anonymised knowledge graph leaks when
the labelling step makes mistakes. It builds a two-key release from FB15k-237,
injects labelling errors at a controlled rate, and attacks every person in it with
ten attackers: six structural similarity indices, three random or structural
controls, and an LLM agent that must cite the graph facts behind every prediction.

## Project Description

A publisher releasing a person-centric knowledge graph masks the facts it judges
sensitive and publishes the rest. Every person then appears under two keys: a
masked key `m_v` carrying the sensitive facts, and an identified key `c_v`
carrying everything else. The release is safe only while nothing links the two.

**Over-masking** is the error where an ordinary, non-sensitive fact is wrongly
labelled sensitive and moved to the masked key. It exposes nothing sensitive by
itself, so no review flags it, but the misfiled fact keeps its links, and those
links can lead back to `c_v`. One misfiled edge can connect the two keys masking
was meant to separate.

The study proceeds in three parts:

**Corpus construction.**

- Every person with a Wikidata label becomes a code `IND_#####`, so an attack must
  answer with a code rather than a name.
- Seven relations are treated as sensitive: six carry special categories of GDPR
  Art. 9(1), and one, net worth, does not. A person's occurrences in those
  relations are rewritten to a second key, per triple rather than per node.
- A MOVE leak then misfiles a fraction `x` of each subject's ordinary facts onto
  their masked key, producing releases at `x = 5%`, `10%` and `15%`, plus corpora
  that fix the leak at exactly 1, 2 or 3 facts per subject.

**Attacks.**

- Six structural indices score candidate person codes through the anchors the two
  keys share: Common Neighbors, Jaccard, Adamic-Adar, Resource Allocation,
  Weighted Anchor and Personalized PageRank.
- An LLM agent retrieves its own evidence and reasons over it across at most five
  rounds, producing a derivation (anchors, candidates, selection, elimination)
  before every answer, and declining when it cannot complete one.
- Three controls bound the scale: global random, two-hop random, and relation-path
  matching, which doubles as a check that no construction stage duplicates edges.

**Analysis.**

- Exposure is modelled against each subject's realised leak and connectivity,
  separated by a within-subject comparison and by the fixed-leak corpora.
- Every answer the agent gets right is audited against four criteria, and
  recomputed deterministically from its own evidence without consulting the model.

## Key Features

**Data construction.**

- Built from FB15k-237 (310,116 triples, three standard splits merged), with
  entities resolved to Wikidata labels.
- A withheld answer key `m_v -> c_v` makes the effect measurable; no public
  release comes with one.
- Reproducible from seed 4242; the fixed-leak path reproduces the proportional
  corpora byte for byte.

**Attack methods.**

- Six parameter-free structural indices, none tuned per corpus.
- A retrieve-then-reason LLM agent extending KG-GPT to several rounds, with a
  bounded evidence budget (150 initial triples, at most 25 new per round).
- Strict top-1 scoring for all ten attackers: the true code must be the single top
  answer, and a tie or a decline counts as a miss.

**Auditing.**

- Four criteria applied to every successful attack: cite only evidence lines and
  graph-internal codes, the predicted code must appear in a cited line, the
  elimination must name a specific contradicting fact, and no real-world name may
  serve as a link.
- A deterministic verifier recomputes each hit from its own stored evidence, so a
  hit can be checked without trusting the model's account of its reasoning.

## Results

At 15% labelling error the best structural index re-identifies 12.52% of subjects,
and its most confident answers are right 64.1% of the time. With correct labelling
no attacker exceeds 0.25%. The agent's row of the headline table:

| Error `x` | Re-identified | Rate (95% CI) |
|---|---|---|
| 0% | 0 / 2,836 | 0.00% [0, 0.14] |
| 5% | 70 / 2,836 | 2.47% [1.96, 3.11] |
| 10% | 100 / 2,836 | 3.53% [2.91, 4.27] |
| 15% | 112 / 2,836 | 3.95% [3.29, 4.73] |

At `x = 10%` one further answer names the true code at rank 2; strict top-1 scores
it as a miss.

Exposure follows how many of a subject's facts were misfiled and, beyond that,
their connectivity; anonymity-set size and the masked attribute show no independent
effect once those are controlled.

## Folder Structure

- `corpus/`: the three stages that turn FB15k-237 into a two-key release.
  - `step1_names_to_id.py`: pseudonymise every person to `IND_#####`.
  - `step2_conditional_mask.py`: move sensitive facts to a second key.
  - `step3_move_leak.py`: simulate the labelling error (`--rate` or `--fixed_k`).
- `deanon_pipeline/`: the retrieve-then-reason attack, its prompts and scoring.
- `baselines/`: the six structural indices and three controls, and the comparison
  restricted to the agent's own evidence.
- `analysis/`: the scripts producing every table and reported figure.
- `run_deanon_attack.py`: entry point for the LLM agent.

## Getting Started

**Prerequisites:**

- Python 3.9+
- NumPy, SciPy, pandas, statsmodels
- `openai` (the client; the attacker model is reached through an
  OpenAI-compatible endpoint) and `python-dotenv`
- An API key for the attacker model. `.env.example` lists the providers the
  pipeline accepts; the paper's runs use `gemini-2.5-flash`.

**Setup:** Clone the repository and install dependencies:

```bash
git clone https://github.com/kienng1608/Field-Level-Anonymisation-Leaks-Quantifying-Re-identification-Risk-from-Over-Masking-in-KG.git
cd Field-Level-Anonymisation-Leaks-Quantifying-Re-identification-Risk-from-Over-Masking-in-KG
pip install -r requirements.txt
cp .env.example .env    # then add your API key
```

**Data Preparation:** Start from the standard FB15k-237 splits, merged into one
graph, then run the three stages. `--rate` takes a fraction (`0.15`), not a
percentage.

```bash
python corpus/step1_names_to_id.py
python corpus/step2_conditional_mask.py
python corpus/step3_move_leak.py --rate 0.05        # misfile 5% of ordinary facts
python corpus/step3_move_leak.py --fixed_k 2        # or exactly 2 facts per subject
```

Stage 1 writes an answer key mapping each `IND_#####` back to the entity it
replaced. **That file is used only for scoring and must not be published**; it is
excluded by `.gitignore`. See *Ethics* below.

**Running the attacks:** the structural indices and controls need no API.

```bash
python baselines/run_non_llm_baselines.py --all_rates   # Table III, all but the agent
python run_deanon_attack.py --corpus <path> --targets all
python baselines/s3_budget_matched.py --rate 15         # indices on the agent's evidence
```

The budget-matched script replays the stored agent runs, so it needs the agent's
results first.

**Reproducing the analyses:** each script writes to `deanon_results/analysis/`,
which is not tracked. The fixed-leak corpora are built first:

```bash
for k in 1 2 3; do
  python corpus/step3_move_leak.py --input_dir data/FB15k-237-id-masked \
         --output_dir data/FB15k-237-id-fixk$k --fixed_k $k
done
```

| Result | Script |
|---|---|
| The agent's row of the headline table | `analysis/extract_paper_data.py` |
| Realised leak against connectivity | `analysis/multivariate_analysis.py` |
| The four audit criteria on every hit | `analysis/audit_15pct_hits.py` |
| Connectivity at a fixed leak | `analysis/fixed_leak.py` |
| Precision of the most confident answers | `analysis/precision_coverage.py` |
| Every attacker on the agent's own evidence | `analysis/equal_evidence_score.py --rate 05/10/15`, then `analysis/equal_evidence_table.py` |
| Precision on equal evidence, split by whether the true code was retrieved | `analysis/equal_evidence_reach.py` |
| The few hits before any labelling error | `analysis/zero_leak_hits.py` |
| Why Personalized PageRank fails | `analysis/ppr_first_step.py`, `analysis/neighbour_exclusion.py` |
| Weighted Anchor's two factors separated | `analysis/weighted_anchor_factors.py` |
| Hits resting on a merged-label node | `analysis/label_merge_hits.py` |
| The release rebuilt without net worth | `analysis/net_worth_sensitivity.py` |
| Whether the evidence forces each agent hit | `analysis/evidence_forced.py` |

Runs use `gemini-2.5-flash`, closed-book, at temperature 0, with corpus seed 4242.
Rates from a language model will not reproduce exactly.

## Ethics

FB15k-237 is a public research dataset whose subjects are public figures and whose
attributes were already public. This repository introduces no new personal data and
reports no re-identified real-world identity: every identifier in the paper and in
these files is a graph-internal code.

The `*_KEY.json` answer keys, which map those codes back to real entities, are
deliberately **not** published. They are needed only to score an attack, never to
run one, and anyone reproducing this work generates their own in stage 1. Example
names appearing inside the prompt templates are illustrative instructions to the
model, not results of any attack.

The attack code is published so that publishers can measure their own releases.
Please use it on data you are authorised to test.

## Citation

If you use this project in your research, please cite the accompanying paper:

```bibtex
@article{nguyen2026overmasking,
  title   = {Field-Level Anonymisation Leaks: Quantifying Re-identification
             Risk from Over-Masking in Knowledge Graphs},
  author  = {Nguyen, Kien and Nguyen, Doan and Nguyen, Quang and Tran, Cong},
  journal = {None},
  year    = {2026}
}
```

## Contact

Posts and Telecommunications Institute of Technology, Ha Noi, Vietnam.
For questions or collaborations, contact Cong Tran (congtt@ptit.edu.vn).
