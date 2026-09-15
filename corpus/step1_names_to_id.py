"""
Step 1 of the dataset pipeline: FB15k-237 (all-MID) -> readable graph with
every PERSON replaced by an opaque ID (IND_0001, ...).

Why this step exists, and why it runs FIRST (before sensitive masking):
  An LLM attacker can shortcut closed-book re-identification by simply
  RECOGNISING a celebrity name in the evidence — "Al Pacino acted in The
  Godfather" needs no graph reasoning if the model already knows who Al
  Pacino is. That shortcut is available to ANY person node the model can
  see, not just the eventual sensitive-relation victims: a bridging node
  (a co-star, a director, a spouse) that still carries a real name lets the
  model recognise ITS way to the target without ever touching the graph
  structure.

  So real names must be gone before sensitive masking runs, and for EVERY
  person in the graph, not just the future victims — masking then applies
  on top of an already-anonymous population, and the only structure left to
  reason over is genuinely structural: which ID is connected to which named
  work (film, award, school - non-person entities keep real names, since an
  attacker in the real world does see those).

Pipeline position:
  FB15k-237 (raw, all-MID)
    -> [this script]      every entity -> readable name; every PERSON -> IND_####
    -> step2_conditional_mask.py   sensitive-position IND_#### -> MID (victims)
    -> step3_move_leak.py          move x% of non-sensitive edges MID-ward

Person detection: identical rule set to pseudonymize_persons.py (a person is
any entity in a person-typed position of any relation) — kept in sync
deliberately so "who counts as a person" never depends on which pipeline
stage is doing the asking.

Usage:
  python codes/step1_names_to_id.py \
      --input_dir data/FB15k-237 \
      --output_dir data/FB15k-237-id-base
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
# Person-typed relation positions — same rule set as pseudonymize_persons.py,
# kept in sync on purpose (see module docstring).
# ============================================================
PERSON_HEAD_PREFIXES = (
    "/people/person/",
    "/base/schemastaging/person_extra/",
    "/people/deceased_person/",
    "/award/award_nominee/",
    "/award/award_winner/",
    "/film/actor/",
    "/film/director/",
    "/film/producer/",
    "/film/writer/",
    "/music/group_member/",
    "/music/artist/",
    "/tv/tv_producer/",
    "/tv/tv_writer/",
    "/organization/organization_founder/",
    "/influence/influence_node/",
    "/celebrities/celebrity/",
    "/sports/pro_athlete/",
    "/government/politician/",
)
PERSON_TAIL_RELATIONS = {
    "/people/cause_of_death/people",
    "/medicine/disease/notable_people_with_this_condition",
    "/people/ethnicity/people",
    "/government/political_party/politicians_in_this_party./government/political_party_tenure/politician",
}
PERSON_TAIL_SUFFIXES = (
    "award_nominee",
    "award_winner",
    "/celebrities/friendship/friend",
    "/celebrities/romantic_relationship/celebrity",
    "/base/popstra/dated/participant",
    "/base/popstra/canoodled/participant",
    "/people/marriage/spouse",
    "/people/sibling_relationship/sibling",
    "/film/performance/actor",
    "/tv/regular_tv_appearance/actor",
    "/music/instrument/instrumentalists",
    "/music/genre/artists",
    "/music/record_label/artist",
    "/education/education/student",
)

# ============================================================
# The 7 sensitive relations from step2_conditional_mask.py, and which side of
# each is the FIXED ATTRIBUTE VALUE (disease name, religion, ethnicity,
# party, currency) as opposed to the victim. Values here must NEVER be
# treated as a person, no matter what PERSON_TAIL_SUFFIXES says.
#
# Why this matters: Freebase has data-quality noise where a relation is
# mis-applied to a non-person MID — e.g. /music/genre/artists sometimes
# points at "Judaism" instead of a musician (Freebase ID /m/EXAMPLE01 has a
# stray genre/artists edge). Without this exclusion, step1 would rename
# "Judaism" itself to an IND_#### code, corrupting every target's religion
# context in step2/step3's evidence. Measured: 225 attribute-value MIDs
# fall on the non-victim side of a sensitive relation somewhere in
# FB15k-237; "Judaism" (/m/EXAMPLE01) is the one instance that also collides
# with a person-typed relation and would otherwise be misclassified.
_SENSITIVE_VICTIM_POSITION = {
    "/medicine/disease/notable_people_with_this_condition":                                       "tail",
    "/people/cause_of_death/people":                                                              "tail",
    "/celebrities/celebrity/sexual_relationships./celebrities/romantic_relationship/celebrity":   "both",
    "/people/person/religion":                                                                    "head",
    "/people/ethnicity/people":                                                                   "tail",
    "/government/political_party/politicians_in_this_party./government/political_party_tenure/politician": "tail",
    "/base/schemastaging/person_extra/net_worth./measurement_unit/dated_money_value/currency":    "head",
}


def find_sensitive_attribute_values(triples):
    """
    MIDs sitting on the fixed-value side of a sensitive relation (a disease,
    a religion, an ethnicity, a party, a currency) — never a person, even if
    they coincidentally satisfy a person-typed relation elsewhere due to a
    Freebase data error.
    """
    attr_values = set()
    for h, r, t in triples:
        pos = _SENSITIVE_VICTIM_POSITION.get(r)
        if pos is None or pos == "both":  # "both" sides are person positions
            continue
        attr_values.add(h if pos == "tail" else t)
    return attr_values


def load_triples(path):
    triples = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples


def load_wiki_mapping(path):
    """freebase_id -> label, first occurrence wins (matches create_anonymized_v2.py)."""
    mid_to_name = {}
    with io.open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            mid, name = parts[0], parts[2]
            if mid not in mid_to_name:
                mid_to_name[mid] = name
    return mid_to_name


def find_persons(triples):
    """Entities appearing in a person-typed position of any relation."""
    persons = set()
    for h, r, t in triples:
        if any(r.startswith(p) for p in PERSON_HEAD_PREFIXES):
            persons.add(h)
        if r in PERSON_TAIL_RELATIONS or r.endswith(PERSON_TAIL_SUFFIXES):
            persons.add(t)
    return persons


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input_dir", default="data/FB15k-237",
                    help="Directory with train/valid/test.txt + fb_wiki_mapping.tsv (all-MID)")
    ap.add_argument("--output_dir", default="data/FB15k-237-id-base",
                    help="Directory to write the ID-based graph")
    ap.add_argument("--mapping", default=None,
                    help="Path to fb_wiki_mapping.tsv (default: <input_dir>/fb_wiki_mapping.tsv)")
    args = ap.parse_args()

    mapping_path = args.mapping or os.path.join(args.input_dir, "fb_wiki_mapping.tsv")
    splits = ["train.txt", "valid.txt", "test.txt"]
    for s in splits:
        p = os.path.join(args.input_dir, s)
        if not os.path.exists(p):
            print(f"[ERROR] Missing input file: {p}")
            sys.exit(1)
    if not os.path.exists(mapping_path):
        print(f"[ERROR] Mapping file not found: {mapping_path}")
        sys.exit(1)

    print(f"Loading wiki mapping from {mapping_path} ...")
    mid_to_name = load_wiki_mapping(mapping_path)
    print(f"  {len(mid_to_name):,} MID -> name entries")

    print("Loading + merging train/valid/test ...")
    all_triples = []
    for s in splits:
        all_triples.extend(load_triples(os.path.join(args.input_dir, s)))
    print(f"  {len(all_triples):,} triples total")

    # ---- MID -> readable name, for every entity that has a label ----
    # Entities with no label (Freebase MIDs fb_wiki_mapping never resolved) stay
    # as raw MIDs — they were always unreadable to a real-world attacker too.
    def to_name(e):
        return mid_to_name.get(e, e)

    named_triples = [(to_name(h), r, to_name(t)) for h, r, t in all_triples]
    unresolved = sum(1 for h, r, t in all_triples
                     if h not in mid_to_name or t not in mid_to_name)
    print(f"  Entities left as raw MID (no wiki label): "
          f"{unresolved:,} triple-endpoints unresolved")

    # ---- Sensitive-attribute values (computed on raw MIDs, then named) ----
    # Must run before person-detection so these never get misclassified —
    # see find_sensitive_attribute_values()'s docstring.
    attr_value_mids = find_sensitive_attribute_values(all_triples)
    attr_value_names = {to_name(m) for m in attr_value_mids}
    print(f"  Sensitive-attribute values (never a person): {len(attr_value_names):,}")

    # ---- Find every PERSON entity (by relation typing, on the NAMED graph) ----
    print("Identifying person entities ...")
    persons = find_persons(named_triples) - attr_value_names
    # Entities still in raw MID form are never renamed to IND_ (they are already
    # opaque, exactly like pseudonymize_persons.py's is_mid() exclusion).
    named_persons = sorted(p for p in persons if not (p.startswith("/m/") and " " not in p))
    print(f"  Person entities found : {len(persons):,}")
    print(f"  Named persons -> IND_ : {len(named_persons):,}")

    code = {name: f"IND_{i:05d}" for i, name in enumerate(named_persons, 1)}

    id_triples = [(code.get(h, h), r, code.get(t, t)) for h, r, t in named_triples]

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "train.txt")
    with io.open(out_path, "w", encoding="utf-8") as f:
        for h, r, t in id_triples:
            f.write(f"{h}\t{r}\t{t}\n")

    # Answer key: IND_#### -> real name. Kept OUTSIDE the graph directory so
    # nothing in the attack pipeline can accidentally load it.
    key_path = os.path.join(os.path.dirname(args.output_dir.rstrip("/\\")) or ".",
                            os.path.basename(args.output_dir) + "_KEY.json")
    json.dump({v: k for k, v in code.items()}, io.open(key_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ---- report ----
    ents = {e for h, _, t in id_triples for e in (h, t)}
    id_ents = {e for e in ents if e.startswith("IND_")}
    mid_ents = {e for e in ents if e.startswith("/m/") and " " not in e}
    print(f"\ntriples          : {len(id_triples):,}")
    print(f"entities         : {len(ents):,}")
    print(f"  IND_#### (renamed persons) : {len(id_ents):,}")
    print(f"  raw MID (unresolved)       : {len(mid_ents):,}")
    print(f"  named (non-person)         : {len(ents) - len(id_ents) - len(mid_ents):,}")
    print(f"\nout : {out_path}")
    print(f"key : {key_path}  (answer key — keep out of the attack pipeline)")


if __name__ == "__main__":
    main()
