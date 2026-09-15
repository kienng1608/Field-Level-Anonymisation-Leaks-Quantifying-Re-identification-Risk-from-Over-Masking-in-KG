"""
Step 1: Target Identification
Scan Hybrid KG for anonymized entities (MIDs) in sensitive relations.
(Adapted from KG-GPT Step 1: Sentence Divide)
"""
import re
from collections import defaultdict

from .config import SENSITIVE_VICTIM_POSITION

MID_PATTERN = re.compile(r'^/m/[a-z0-9_]+$')


def is_mid(entity):
    """Check if entity is an anonymized MID (e.g., /m/EXAMPLE01)."""
    return bool(MID_PATTERN.match(entity))


def _victim_positions(relation, victim_position_map):
    """
    Positions ('head'/'tail') holding the anonymized PERSON for `relation`.

    Falls back to BOTH endpoints when the relation is absent from the map, so a
    custom sensitive relation is never silently skipped — but the known 7 are
    all mapped, which keeps non-person endpoints (religion / currency / ethnicity
    nodes) from being mistaken for targets.
    """
    pos = victim_position_map.get(relation, "both")
    if pos == "both":
        return ("head", "tail")
    if pos == "none":
        return ()
    return (pos,)


def identify_targets(triples, sensitive_relations, victim_position_map=None):
    """
    Scan all triples to find anonymized PERSON MIDs in sensitive relations.

    A MID is recorded as a target ONLY when it sits at the victim (person)
    position for its relation — so the MID of a non-person endpoint (e.g. a
    religion in `/people/person/religion`, or a currency in net_worth) is NOT
    collected as a de-anonymization target.

    Args:
        triples: list of (head, relation, tail)
        sensitive_relations: relations considered sensitive
        victim_position_map: dict relation -> "head"|"tail"|"both"|"none".
            Defaults to config.SENSITIVE_VICTIM_POSITION.

    Returns:
        dict: {
            mid: {
                "mid": str,
                "sensitive_contexts": [
                    {"relation": str, "context_entity": str, "position": "head"|"tail"}
                ]
            }
        }
    """
    if victim_position_map is None:
        victim_position_map = SENSITIVE_VICTIM_POSITION

    targets = {}

    for h, r, t in triples:
        if r not in sensitive_relations:
            continue

        vpos = _victim_positions(r, victim_position_map)

        # Tail is the anonymized person
        if "tail" in vpos and is_mid(t):
            if t not in targets:
                targets[t] = {"mid": t, "sensitive_contexts": []}
            targets[t]["sensitive_contexts"].append({
                "relation": r,
                "context_entity": h,
                "position": "tail",
            })

        # Head is the anonymized person
        if "head" in vpos and is_mid(h):
            if h not in targets:
                targets[h] = {"mid": h, "sensitive_contexts": []}
            targets[h]["sensitive_contexts"].append({
                "relation": r,
                "context_entity": t,
                "position": "head",
            })

    return targets


def select_targets(targets, KG, wiki_mapping, sensitive_relations, n_samples=10,
                   target_relations=None):
    """
    Select high-value targets for attack demo.

    Args:
        targets: dict from identify_targets()
        KG: dict, KG[entity][relation] = [tails]
        wiki_mapping: dict, MID -> real name
        sensitive_relations: list of relations considered sensitive
        n_samples: int, how many targets to return (after sorting)
        target_relations: optional list/set of relations to restrict targets to.
            None (default) = no restriction (attack ALL targets that have ground truth).
            Pass e.g. ["/people/cause_of_death/people"] to focus on that relation.

    Criteria:
      - Must have ground truth in wiki_mapping
      - Optionally restricted to targets touching `target_relations`
      - Sorted by number of sensitive relations (more context = better signal)
    """
    target_list = []
    target_rel_set = set(target_relations) if target_relations else None

    for mid, info in targets.items():
        if target_rel_set is not None:
            has_target_rel = any(
                ctx["relation"] in target_rel_set
                or any(ctx["relation"].endswith(tr) for tr in target_rel_set)
                for ctx in info["sensitive_contexts"]
            )
            if not has_target_rel:
                continue

        # Public relations: look up by real name (non-sensitive triples use the
        # real name as key in KG, while sensitive triples use the MID key).
        real_name_key = wiki_mapping.get(mid, "")
        all_rels = set(KG.get(mid, {}).keys()) | set(KG.get(real_name_key, {}).keys())
        public_rels = [r for r in all_rels if r not in sensitive_relations
                       and r.lstrip("~") not in sensitive_relations]
        n_public = len(public_rels)

        # Đếm số lượng bối cảnh nhạy cảm
        n_sensitive = len(info["sensitive_contexts"])

        # Must have ground truth
        real_name = wiki_mapping.get(mid)
        if not real_name:
            continue

        target_list.append({
            "mid": mid,
            "real_name": real_name,
            "n_public_relations": n_public,
            "n_sensitive": n_sensitive,
            "sensitive_contexts": info["sensitive_contexts"],
        })

    # Sort by amount of sensitive context available
    target_list.sort(key=lambda x: x["n_sensitive"], reverse=True)

    return target_list[:n_samples]
