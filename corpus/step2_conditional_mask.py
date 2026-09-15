"""
Step 2 of the dataset pipeline: conditional per-triple sensitive masking,
run on top of the ID-based graph produced by step1_names_to_id.py.

What changes from the original create_anonymized_v2.py:
  The old script masked victims straight from their REAL NAME to a MID. This
  version masks from IND_#### (their step-1 identity code) to a MID instead —
  the victim's non-sensitive identity is a code, not a recognisable name, so
  an LLM attacker never gets to shortcut via memorised celebrity knowledge for
  ANY person in the graph, victim or bystander.

Mechanism (same conditional-masking idea as before):
  - In SENSITIVE triples: the victim's IND_#### is replaced by their
    Freebase MID (/m/xxxxx) — this IS the anonymization.
  - In NON-SENSITIVE triples: the victim keeps their IND_#### code.
  A victim therefore ends up with TWO keys in the graph: MID (sensitive
  triples) and IND_#### (non-sensitive triples) — structurally disjoint,
  exactly like the real-name version was, just with the "name" side replaced
  by a code.

Pipeline position:
  FB15k-237-id-base (from step1)
    -> [this script]     victim's IND_#### -> MID, sensitive positions ONLY
    -> step3_move_leak.py

Usage:
  python codes/step2_conditional_mask.py \
      --input_dir data/FB15k-237-id-base \
      --output_dir data/FB15k-237-id-masked
"""
import argparse
import collections
import io
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ============================================================
# 7 Sensitive Relations — must stay identical to
# deanon_pipeline/config.py::SENSITIVE_RELATIONS and the original
# codes/create_anonymized_v2.py.
# ============================================================
SENSITIVE_RELATIONS = [
    "/medicine/disease/notable_people_with_this_condition",
    "/people/cause_of_death/people",
    "/celebrities/celebrity/sexual_relationships./celebrities/romantic_relationship/celebrity",
    "/people/person/religion",
    "/people/ethnicity/people",
    "/government/political_party/politicians_in_this_party./government/political_party_tenure/politician",
    "/base/schemastaging/person_extra/net_worth./measurement_unit/dated_money_value/currency",
]

SENSITIVE_VICTIM_POSITION = {
    "/medicine/disease/notable_people_with_this_condition":                                       "tail",
    "/people/cause_of_death/people":                                                              "tail",
    "/celebrities/celebrity/sexual_relationships./celebrities/romantic_relationship/celebrity":   "both",
    "/people/person/religion":                                                                    "head",
    "/people/ethnicity/people":                                                                   "tail",
    "/government/political_party/politicians_in_this_party./government/political_party_tenure/politician": "tail",
    "/base/schemastaging/person_extra/net_worth./measurement_unit/dated_money_value/currency":    "head",
}


def load_triples(path):
    triples = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples


def is_ind_code(e):
    return e.startswith("IND_")


def build_victim_mid_map(triples, key_path):
    """
    Assign each victim IND_#### a Freebase MID.

    step1's answer key (IND_KEY.json: IND_#### -> real name) does NOT carry
    the person's original Freebase MID, so we recover it from
    fb_wiki_mapping.tsv by looking up which MID maps to that real name — same
    source table step1 used, just inverted.
    """
    from step1_names_to_id import load_wiki_mapping  # sibling module

    ind_to_name = json.load(io.open(key_path, encoding="utf-8"))
    mid_to_name = load_wiki_mapping(
        os.path.join("data", "FB15k-237", "fb_wiki_mapping.tsv"))
    name_to_mid = {}
    for mid, name in mid_to_name.items():
        name_to_mid.setdefault(name, mid)  # first MID wins, mirrors step1

    victim_mid = {}
    for ind, name in ind_to_name.items():
        mid = name_to_mid.get(name)
        if mid:
            victim_mid[ind] = mid
    return victim_mid, ind_to_name


def identify_victims(triples, sensitive_relations, victim_position_map):
    """Which IND_#### codes sit at the victim position of a sensitive relation."""
    victims = set()
    for h, r, t in triples:
        pos = victim_position_map.get(r)
        if pos is None or pos == "none":
            continue
        if pos in ("head", "both") and is_ind_code(h):
            victims.add(h)
        if pos in ("tail", "both") and is_ind_code(t):
            victims.add(t)
    return victims


def anonymize_triples(triples, victim_ind_to_mid, victim_position_map):
    """Conditional per-triple masking: victim IND_#### -> MID at sensitive positions only."""
    anonymized = []
    stats = collections.Counter()

    for h, r, t in triples:
        pos = victim_position_map.get(r)
        if pos is None:
            anonymized.append((h, r, t))
            stats["non_sensitive"] += 1
            continue
        if pos == "none":
            anonymized.append((h, r, t))
            stats["sensitive_implicit"] += 1
            continue

        h_out = victim_ind_to_mid.get(h, h) if pos in ("head", "both") else h
        t_out = victim_ind_to_mid.get(t, t) if pos in ("tail", "both") else t

        if h_out != h or t_out != t:
            stats["sensitive_anonymized"] += 1
        else:
            stats["sensitive_unchanged"] += 1
        anonymized.append((h_out, r, t_out))

    return anonymized, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input_dir", default="data/FB15k-237-id-base",
                    help="Output directory of step1_names_to_id.py")
    ap.add_argument("--output_dir", default="data/FB15k-237-id-masked",
                    help="Directory to write the masked graph")
    ap.add_argument("--key", default=None,
                    help="Path to step1's IND_KEY.json (default: sibling of --input_dir)")
    args = ap.parse_args()

    key_path = args.key or (args.input_dir.rstrip("/\\") + "_KEY.json")
    if not os.path.exists(key_path):
        print(f"[ERROR] step1 answer key not found: {key_path}")
        print("        Run step1_names_to_id.py first.")
        sys.exit(1)

    train_path = os.path.join(args.input_dir, "train.txt")
    if not os.path.exists(train_path):
        print(f"[ERROR] Missing input file: {train_path}")
        sys.exit(1)

    print(f"Loading triples from {train_path} ...")
    triples = load_triples(train_path)
    print(f"  {len(triples):,} triples")

    print("Recovering victim MIDs from fb_wiki_mapping.tsv ...")
    victim_mid_all, ind_to_name = build_victim_mid_map(triples, key_path)
    print(f"  {len(victim_mid_all):,} IND_#### codes have a recoverable Freebase MID")

    print("Identifying victims (IND_#### at a sensitive-relation victim position) ...")
    victims = identify_victims(triples, SENSITIVE_RELATIONS, SENSITIVE_VICTIM_POSITION)
    print(f"  {len(victims):,} victims found")

    # Only victims that also have a resolvable MID can be masked; the few that
    # don't (label collision, missing wiki entry) are logged, not silently kept.
    victim_ind_to_mid = {v: victim_mid_all[v] for v in victims if v in victim_mid_all}
    unresolved_victims = victims - set(victim_ind_to_mid)
    if unresolved_victims:
        print(f"  [WARN] {len(unresolved_victims):,} victims have no recoverable MID "
              f"— left as IND_#### even at sensitive positions")

    print("Applying conditional masking ...")
    anonymized, stats = anonymize_triples(triples, victim_ind_to_mid, SENSITIVE_VICTIM_POSITION)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "train.txt")
    with io.open(out_path, "w", encoding="utf-8") as f:
        for h, r, t in anonymized:
            f.write(f"{h}\t{r}\t{t}\n")

    # victims.json: MID -> IND_#### (the "ground truth" the attack has to
    # recover, in this graph's own vocabulary — matches how
    # run_deanon_attack.py already prefers a dataset's own victims.json over
    # wiki_mapping's real names).
    victims_path = os.path.join(args.output_dir, "victims.json")
    mid_to_ind = {mid: ind for ind, mid in victim_ind_to_mid.items()}
    json.dump(mid_to_ind, io.open(victims_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"\nSummary:")
    print(f"  Sensitive anonymized      : {stats['sensitive_anonymized']:,}")
    print(f"  Sensitive no victim found : {stats['sensitive_unchanged']:,}")
    print(f"  Sensitive implicit person : {stats['sensitive_implicit']:,}")
    print(f"  Non-sensitive (unchanged) : {stats['non_sensitive']:,}")
    print(f"  Victims manifest          : {victims_path}  ({len(mid_to_ind):,} victims)")
    print(f"  Output directory          : {args.output_dir}")


if __name__ == "__main__":
    main()
