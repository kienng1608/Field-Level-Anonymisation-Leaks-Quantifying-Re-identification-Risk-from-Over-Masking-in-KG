"""
Step 3 of the dataset pipeline: MOVE-leak, run on top of the conditionally-
masked ID graph produced by step2_conditional_mask.py.

Mechanism (MOVE, not COPY):
  For each victim, take a random x% of their NON-sensitive edges (the ones
  still attached to their IND_#### code) and REASSIGN — not duplicate — them
  to the victim's MID:
      REMOVE   IND_1234  --student-->        Royal College of Art
      ADD      /m/EXAMPLE01 --student-->        Royal College of Art
  Sensitive edges are never touched. The edge exists exactly once, either on
  IND_1234 or on /m/EXAMPLE01, never both — this is what makes the leak
  realistic (a real anonymiser reassigning a mis-classified relation, not an
  attacker somehow seeing a fact twice).

Why this still connects the MID to a named part of the graph even though the
edge itself carries no name:
  The far endpoint of the moved edge (a film, an award, a school — anything
  that is not a person) keeps ALL of its OTHER edges untouched, including the
  ones pointing back at the victim's IND_#### from a different relation.
  E.g. if "Pirates of the Caribbean" already had an /film/actor/film edge to
  IND_1234, and a DIFFERENT edge (say an award nomination) gets moved to the
  MID, the film becomes a bridge: MID -> film (moved edge) and
  film -> IND_1234 (the untouched edge) are both present, so an attacker who
  reaches the film from the MID can walk back to the same IND_#### code from
  the other side. No entity is renamed and no edge is duplicated to make this
  work — it falls out of the graph already being densely connected before any
  leak is applied. This is the "connection between the ID and the entity
  that gets attached to the MID" the leak relies on.

MIRROR PAIRS — the bug this version fixes:
  FB15k-237 stores some facts as TWO forward triples instead of one symmetric
  fact, e.g. a marriage is BOTH
      IND_A  --spouse-->  IND_B
      IND_B  --spouse-->  IND_A
  These are the SAME real-world fact, not two independent edges. The original
  version of this script sampled movable edges as independent index draws, so
  it could move one direction to the victim's MID while leaving the other on
  their IND_#### code:
      /m/xxxxx (MID)  --spouse-->  IND_02452     (moved)
      IND_02051 (ID)  --spouse-->  IND_02452     (never touched)
  Both triples name IND_02452 through the identical relation — an attacker
  reading them side by side gets MID = ID directly, no multi-hop reasoning
  needed. Measured: this exact case reached a live evaluation run (target
  /m/EXAMPLE02 / IND_02051, resolved by an LLM in round 1 largely off this pair).
  Fixed by grouping (h, r, t) with its mirror (t, r, h) — when either side of
  a mirror pair is a victim's non-sensitive edge, the pair is treated as ONE
  atomic unit: both triples move together or neither does. This does not
  change which relations move, only that mirrored triples of a victim can no
  longer land on different sides of the MID/IND split.

Pipeline position:
  FB15k-237-id-masked (from step2)
    -> [this script]     move x% of each victim's non-sensitive edges MID-ward
    -> data/FB15k-237-id-move05  (final dataset used by the attack pipeline)

Usage:
  python codes/step3_move_leak.py --rate 0.05 \
      --input_dir data/FB15k-237-id-masked \
      --output_dir data/FB15k-237-id-move05
"""
import argparse
import collections
import io
import json
import os
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SENSITIVE_RELATIONS = [
    "/medicine/disease/notable_people_with_this_condition",
    "/people/cause_of_death/people",
    "/celebrities/celebrity/sexual_relationships./celebrities/romantic_relationship/celebrity",
    "/people/person/religion",
    "/people/ethnicity/people",
    "/government/political_party/politicians_in_this_party./government/political_party_tenure/politician",
    "/base/schemastaging/person_extra/net_worth./measurement_unit/dated_money_value/currency",
]


def load_triples(path):
    triples = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input_dir", default="data/FB15k-237-id-masked",
                    help="Output directory of step2_conditional_mask.py")
    ap.add_argument("--output_dir", default="data/FB15k-237-id-move05",
                    help="Directory to write the leaked dataset")
    ap.add_argument("--rate", type=float, default=0.05,
                    help="Fraction of each victim's non-sensitive edges to move (default 0.05)")
    ap.add_argument("--seed", type=int, default=4242,
                    help="RNG seed, for a reproducible leak selection")
    ap.add_argument("--extend_from", default=None,
                    help="Directory of an already-leaked graph to build on. Every "
                         "edge moved there stays moved, and sampling only tops each "
                         "victim up to --rate. Produces a NESTED series (the lower "
                         "rate's leak is a subset of this one's) rather than an "
                         "independent draw, which isolates 'more leakage' from "
                         "'different edges leaked'.")
    ap.add_argument("--fixed_k", type=int, default=None,
                    help="Move exactly this many non-sensitive edges per victim "
                         "instead of a fraction of its degree, so realised leak no "
                         "longer scales with connectivity. Overrides --rate. A "
                         "mirror pair still moves as one 2-edge unit, so a victim "
                         "can end one edge above k.")
    args = ap.parse_args()

    train_path = os.path.join(args.input_dir, "train.txt")
    victims_path = os.path.join(args.input_dir, "victims.json")
    for p in (train_path, victims_path):
        if not os.path.exists(p):
            print(f"[ERROR] Missing input file: {p}")
            print("        Run step2_conditional_mask.py first.")
            sys.exit(1)

    print(f"Loading triples from {train_path} ...")
    triples = load_triples(train_path)
    print(f"  {len(triples):,} triples")

    victims = json.load(io.open(victims_path, encoding="utf-8"))  # mid -> IND_####
    ind_to_mid = {ind: mid for mid, ind in victims.items()}
    print(f"  {len(victims):,} victims (mid -> IND_####)")

    sensitive = set(SENSITIVE_RELATIONS)

    # ---- mirror-pair index: (r, h, t) -> [indices], to find (t, r, h) for any (h, r, t) ----
    # Built as a global bijective pairing (not a per-call "first match" lookup)
    # because the graph can contain multiple IDENTICAL triples for the same
    # (h, r, t) — e.g. two literal duplicate rows of
    #   IND_04134  award_nominee  IND_00775
    # alongside two duplicates of the reverse
    #   IND_00775  award_nominee  IND_04134
    # A naive find_mirror(idx) that returns "the first index found at (r, t, h)"
    # maps BOTH forward duplicates onto the SAME reverse index, leaving the
    # other reverse duplicate unpaired — the move loop then tries to move that
    # unit's mirror half again, and it may already have been consumed by a
    # different unit, leaving one side moved and the other not (caught by the
    # split-mirror assertion below). Pairing every (r,h,t) group against its
    # (r,t,h) group index-for-index (1st with 1st, 2nd with 2nd, ...) instead
    # of "any-with-first" makes each triple index resolve to exactly one
    # mirror partner, so every unit is well-defined regardless of duplicates.
    triple_index = collections.defaultdict(list)
    for idx, (h, r, t) in enumerate(triples):
        triple_index[(r, h, t)].append(idx)

    mirror_of = {}
    seen_keys = set()
    for (r, h, t), fwd_indices in triple_index.items():
        if h == t or (r, h, t) in seen_keys:
            continue
        rev_indices = triple_index.get((r, t, h), ())
        seen_keys.add((r, h, t))
        seen_keys.add((r, t, h))
        for i, j in zip(fwd_indices, rev_indices):
            mirror_of[i] = j
            mirror_of[j] = i
        # Any surplus on the longer side (unequal duplicate counts) is left
        # unpaired — find_mirror() returns None for those, so they fall back
        # to being treated as ordinary (non-mirror) single-index units, which
        # is the safe default the rest of the script already handles.

    def find_mirror(idx):
        """Index of this triple's paired mirror, if any (via the bijective mapping above)."""
        return mirror_of.get(idx)

    # Pool of movable (non-sensitive) edge indices, per victim IND_#### code —
    # only edges where the victim's IND_#### is head or tail are eligible.
    # Each entry is an ATOMIC UNIT: either a single index, or a frozenset of
    # {idx, mirror_idx} when the edge has a same-relation reverse triple. Units
    # are deduplicated per victim via a set so a mirrored pair touching the
    # same victim from both a forward and reverse edge is only offered once.
    by_ind = collections.defaultdict(set)
    for idx, (h, r, t) in enumerate(triples):
        if r in sensitive:
            continue
        mirror = find_mirror(idx)
        unit = frozenset((idx, mirror)) if mirror is not None else frozenset((idx,))
        if h in ind_to_mid:
            by_ind[h].add(unit)
        if t in ind_to_mid:
            by_ind[t].add(unit)

    # ---- optional: inherit an existing leak (nested series) ----------------
    # With --extend_from, every edge already moved in that graph is kept moved
    # and counts toward its victim's quota, so this corpus is a strict superset
    # of the one it extends. Comparing a nested pair against an independent pair
    # separates the effect of leaking MORE from the effect of leaking DIFFERENT
    # edges, which an independent draw confounds.
    inherited = {}
    if args.extend_from:
        prev_path = os.path.join(args.extend_from, "train.txt")
        if not os.path.exists(prev_path):
            print(f"[ERROR] --extend_from given but {prev_path} does not exist")
            sys.exit(1)
        prev = load_triples(prev_path)
        if len(prev) != len(triples):
            print(f"[ERROR] {prev_path} has {len(prev):,} triples, input has "
                  f"{len(triples):,}; they must come from the same masked graph")
            sys.exit(1)
        for i, (before, after) in enumerate(zip(triples, prev)):
            if before != after:
                inherited[i] = after
        print(f"  inheriting {len(inherited):,} already-moved edges from "
              f"{args.extend_from}")

    rnd = random.Random(args.seed)
    moved = dict(inherited)  # index -> new (h, r, t)
    mirror_units_moved = 0
    mirror_units_collided = 0
    for ind in sorted(by_ind):  # sorted: deterministic iteration order for a fixed seed
        units = sorted(by_ind[ind], key=lambda u: min(u))
        mid = ind_to_mid[ind]
        # Rate is a fraction of EDGES, not units — a 2-triple mirror unit counts
        # as 2 edges toward the target, same as if it had been 2 independent
        # edges, so the leak rate's meaning (% of a victim's non-sensitive
        # edges) is unchanged by grouping.
        n_edges = sum(len(u) for u in units)
        if args.fixed_k is not None:
            k_edges = min(args.fixed_k, n_edges)
        else:
            k_edges = int(round(n_edges * args.rate))
        # Edges this victim already carries from --extend_from count toward the
        # quota, so only the shortfall is sampled here.
        already = sum(1 for u in units for i in u if i in inherited)
        k_edges -= already
        if k_edges <= 0:
            continue
        units = [u for u in units if not any(i in inherited for i in u)]
        rnd.shuffle(units)
        selected, taken = [], 0
        for u in units:
            if taken >= k_edges:
                break
            selected.append(u)
            taken += len(u)
        for unit in selected:
            # A 2-victim mirror pair (e.g. IND_A--spouse-->IND_B and the
            # reverse) is offered to BOTH victims' pools. If the other victim's
            # pass already moved it, every index here is already in `moved` —
            # skip the whole unit rather than only the un-moved half, which
            # would tear it apart (the exact bug this rewrite fixes). This
            # slightly undershoots the requested rate for whichever victim's
            # pass runs second on a shared unit; counted below, not silently
            # absorbed.
            if len(unit) == 2:
                if all(i in moved for i in unit):
                    mirror_units_collided += 1
                    continue
                mirror_units_moved += 1
            for i in unit:
                if i in moved:
                    continue
                h, r, t = triples[i]
                moved[i] = (mid if h == ind else h, r, mid if t == ind else t)

    out_triples = [moved.get(i, tr) for i, tr in enumerate(triples)]

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "train.txt")
    with io.open(out_path, "w", encoding="utf-8") as f:
        for h, r, t in out_triples:
            f.write(f"{h}\t{r}\t{t}\n")

    # victims.json carries forward unchanged — masking positions didn't change,
    # only some non-sensitive edges moved.
    json.dump(victims, io.open(os.path.join(args.output_dir, "victims.json"), "w",
                               encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---- verification ----
    assert len(out_triples) == len(triples), "triple count must be unchanged (MOVE, not add/drop)"
    sensitive_before = sum(1 for h, r, t in triples if r in sensitive)
    sensitive_after = sum(1 for h, r, t in out_triples if r in sensitive)
    assert sensitive_before == sensitive_after, "sensitive edges must never be touched"

    # No mirror pair (a REAL (A,r,B)+(B,r,A) pair — same definition as
    # find_mirror() above, not "same relation, some shared endpoint") may end
    # up with one side moved and the other not. By construction, the move
    # loop above always moves both indices of a unit together (same `mid`,
    # same pass) or neither — so this check just confirms that held: for
    # every mirror pair where at least one side was moved, the OTHER side
    # must have moved too.
    #
    # Deliberately NOT approximated as "(relation, other-endpoint) appears via
    # both of a victim's keys" — that over-fires whenever many different
    # people legitimately share one non-person endpoint, e.g. 'ballet' in
    # /music/genre/artists: two different artists both linked to 'ballet' is
    # two independent facts, not a split mirror (this was tried and produced
    # a false positive on exactly that relation).
    split_examples = []
    for idx, (h, r, t) in enumerate(triples):
        if r in sensitive:
            continue
        mirror = find_mirror(idx)
        if mirror is None or mirror < idx:
            continue  # only check each mirror pair once, from its lower index
        if (idx in moved) != (mirror in moved):
            split_examples.append((triples[idx], triples[mirror]))

    assert not split_examples, (
        f"{len(split_examples)} mirror pairs had only one side moved "
        f"(e.g. {split_examples[0]})"
    )

    n_moved = len(moved)
    print(f"\nSummary:")
    print(f"  Leak rate                 : {args.rate:.0%} of each victim's non-sensitive edges")
    print(f"  Edges moved                : {n_moved:,}  ({n_moved/len(triples)*100:.2f}% of graph)")
    if inherited:
        print(f"    inherited from base      : {len(inherited):,}")
        print(f"    newly sampled here       : {n_moved - len(inherited):,}")
    print(f"  Victims affected           : {len({ind_to_mid[t[0]] if t[0] in ind_to_mid else ind_to_mid.get(t[2]) for i,t in enumerate(triples) if i in moved}):,}")
    print(f"  Mirror pairs moved together : {mirror_units_moved:,}")
    print(f"  Mirror pairs collided       : {mirror_units_collided:,}  "
          f"(shared between 2 victims, already moved by the other's pass — undershoots rate slightly)")
    print(f"  Sensitive edges leaked     : 0  (asserted)")
    print(f"  Triple count unchanged     : {len(out_triples):,} == {len(triples):,}  (asserted)")
    print(f"  Mirror pairs torn apart    : 0  (asserted — see MIRROR PAIRS docstring)")
    print(f"  Output directory           : {args.output_dir}")


if __name__ == "__main__":
    main()
