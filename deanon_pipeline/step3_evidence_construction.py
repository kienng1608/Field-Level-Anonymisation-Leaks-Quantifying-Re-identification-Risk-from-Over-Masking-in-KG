"""
Step 3: Evidence Graph Construction
Build evidence subgraph using KG-GPT's structured query approach:
  Pha A: Create (entity × relation) query patterns
  Pha B: Verify patterns against KG → confirmed triples
  Pha C: Multi-hop join → connected subgraph (with adaptive hub filter
         and person-only restriction)
  Pha D: Score + deduplicate + truncate by IDF-based ranking
"""
from collections import defaultdict, deque
from .step1_target_identification import is_mid
from .config import (
    SENSITIVE_RELATIONS, MAX_EVIDENCE_TRIPLES,
    EVIDENCE_ALPHA, EVIDENCE_BETA, EVIDENCE_GAMMA, PERSON_ONLY_MULTIHOP,
    SKIP_MID_IN_EXPANSION,
    EVIDENCE_PROXIMITY, EVIDENCE_MAX_HOP_KEEP, EVIDENCE_REQUIRE_CONNECTED,
    EVIDENCE_SELECTION,
)


def _is_person_likely(entity, kg_stats):
    """Entity passes person-likely filter if it's a MID or appears in /people/... relations."""
    if is_mid(entity):
        return True
    if kg_stats and entity in kg_stats.get("person_entities", set()):
        return True
    return False


def compute_placeholder_fingerprint(KG, mid, target_mid, placeholder_map=None, kg_stats=None):
    """
    Compute structural metadata for a placeholder MID that helps the LLM
    hypothesize who this anonymized person is.

    Returns dict with:
      cluster_size:      number of other MIDs sharing sibling/spouse/colleague ties
      anchor_neighbors:  list of named entities directly connected (strings, not MIDs)
      degree:            total number of connections in KG
      rarity_score:      1/sharing for the rarest (rel, value) pair, or 0
      role_hints:        list of relation verbs connecting this MID to TARGET
      sibling_count:     count of sibling MIDs (excludes target)
      spouse_count:      count of spouse MIDs

    Uses only structural info in the anonymized KG — no wiki_mapping.
    """
    if KG is None or mid not in KG:
        return {
            "cluster_size": 0, "anchor_neighbors": [], "degree": 0,
            "rarity_score": 0.0, "role_hints": [],
            "sibling_count": 0, "spouse_count": 0,
        }

    sel = (kg_stats or {}).get("relation_selectivity", {})

    cluster_mids   = set()
    anchor_names   = []
    role_hints     = set()
    sibling_count  = 0
    spouse_count   = 0
    best_rarity    = 0.0
    degree         = 0

    for rel, tails in KG[mid].items():
        actual_rel = rel.lstrip("~")
        degree += len(tails)

        for tail in tails:
            if tail == target_mid:
                # Direct connection to target — capture role
                role_hints.add(_short_role(actual_rel))
            elif is_mid(tail):
                cluster_mids.add(tail)
            else:
                if tail not in anchor_names:
                    anchor_names.append(tail)
                # Compute rarity for this (rel, value) pair
                sharing = sum(1 for e in KG if rel in KG[e] and tail in KG[e][rel])
                if sharing > 0:
                    rarity = 1.0 / sharing
                    if rarity > best_rarity:
                        best_rarity = rarity

            if "sibling" in actual_rel:
                if tail != target_mid:
                    sibling_count += 1
            if "spouse" in actual_rel:
                if tail != target_mid:
                    spouse_count += 1

    return {
        "cluster_size": len(cluster_mids),
        "anchor_neighbors": anchor_names[:5],
        "degree": degree,
        "rarity_score": best_rarity,
        "role_hints": sorted(role_hints),
        "sibling_count": sibling_count,
        "spouse_count": spouse_count,
    }


def _short_role(relation):
    """Map a sensitive relation path to a short human-readable role token."""
    if "sibling" in relation:
        return "sibling"
    if "spouse" in relation:
        return "spouse"
    if "cause_of_death" in relation:
        return "cause_of_death"
    if "employment_history" in relation:
        return "employer"
    if "job_title" in relation:
        return "job_title"
    if "net_worth" in relation:
        return "net_worth"
    if "notable_people_with_this_condition" in relation:
        return "disease"
    return relation.split("/")[-1] or "related"


def construct_evidence_graph(KG, target_mid, sensitive_contexts, selected_relations,
                             n_hops=2, max_triples=None, kg_stats=None):
    """
    Build evidence subgraph using KG-GPT style.

    Args:
        KG: dict, KG[entity][relation] = [tails]
        target_mid: str, the anonymized entity MID
        sensitive_contexts: list of dicts, sensitive context anchors
        selected_relations: list, top-K relations from Step 2
        n_hops: int, number of expansion hops
        max_triples: int, max evidence triples
        kg_stats: dict from data_loader.compute_kg_stats(); when supplied,
            enables adaptive hub filter, IDF scoring, and person-only multi-hop.
            If None, falls back to legacy heuristics (constant hub threshold,
            unscored deduplication) for backwards compatibility.

    Returns:
        final_evidence: list of [head, relation, tail] triples
    """
    if max_triples is None:
        max_triples = MAX_EVIDENCE_TRIPLES

    hub_threshold = kg_stats["hub_threshold"] if kg_stats else 500.0
    relation_idf = kg_stats["relation_idf"] if kg_stats else {}
    person_entities = kg_stats["person_entities"] if kg_stats else set()

    selected_set = set(selected_relations)
    context_entities = {ctx["context_entity"] for ctx in sensitive_contexts}

    # ─── PHA A: Create query patterns ─────────────────────
    anchors = [target_mid] + [ctx["context_entity"] for ctx in sensitive_contexts]

    total_patterns = []
    for anchor in set(anchors):
        for rel in selected_relations:
            total_patterns.append([anchor, rel])
            if not rel.startswith("~"):
                total_patterns.append([anchor, "~" + rel])
            else:
                total_patterns.append([anchor, rel[1:]])

    # ─── PHA B: Verify patterns against KG ────────────────
    confirmed_triples = []
    discovered_entities = set()

    for pattern in total_patterns:
        entity = pattern[0]
        rel = pattern[1]

        if entity not in KG or rel not in KG[entity]:
            continue

        tails = KG[entity][rel]
        for tail in tails:
            other = tail
            if rel.startswith("~"):
                triple = [tail, rel[1:], entity]
            else:
                triple = [entity, rel, tail]

            # A fact about some OTHER anonymized person names nobody and cannot be
            # followed anywhere useful, so it only consumes budget. Facts on the
            # target's own MID are kept — they are the problem statement.
            if (SKIP_MID_IN_EXPANSION and is_mid(other)
                    and other != target_mid and entity != target_mid):
                continue

            confirmed_triples.append(triple)
            discovered_entities.add(other)

    # ─── PHA C: Multi-hop join (adaptive hub + optional person-only) ─
    # NOTE: this used to run a SINGLE expansion pass regardless of `n_hops`, so the
    # subgraph never reached beyond 2 hops from the target. Measured on the 274
    # clean-path victims, 78% of real names sit exactly 3 hops away — they were
    # structurally unreachable. The loop below now actually iterates `n_hops` times,
    # expanding the frontier discovered by the previous round.
    multihop_triples = []

    frontier = set(discovered_entities)
    visited_expanders = set()

    # Cap how many nodes we expand per hop. Without this the frontier explodes
    # combinatorially (a single 3-hop target took 15s and produced tens of
    # thousands of candidate triples). Expanding the LOWEST-degree nodes first
    # keeps the most selective — i.e. most identifying — branches.
    MAX_FRONTIER_PER_HOP = 400

    for _hop in range(max(0, n_hops - 1)):
        if not frontier:
            break
        if len(frontier) > MAX_FRONTIER_PER_HOP:
            frontier = set(sorted(
                frontier,
                key=lambda e: sum(len(v) for v in KG[e].values()) if e in KG else 0,
            )[:MAX_FRONTIER_PER_HOP])
        next_frontier = set()

        for disc_entity in frontier:
            if disc_entity in visited_expanders or disc_entity not in KG:
                continue
            visited_expanders.add(disc_entity)

            # Adaptive hub filter (replaces hard-coded 500 constant)
            total_connections = sum(len(tails) for tails in KG[disc_entity].values())
            if total_connections > hub_threshold:
                continue

            # Person-only restriction: only expand through anonymized MID entities.
            # String-label entities (e.g. "Charlie Chaplin", "Douglas Fairbanks")
            # discovered transitively via a company anchor are kept as 1-hop facts
            # but must NOT be used as expansion hubs — that would flood evidence with
            # their full public profiles, burying the actual target's signal.
            if PERSON_ONLY_MULTIHOP and kg_stats:
                if not (is_mid(disc_entity) and _is_person_likely(disc_entity, kg_stats)):
                    continue

            # Never expand THROUGH another anonymized MID. Their own facts are the
            # generic attributes already barred as grounding, so the chain just
            # produces more un-nameable MIDs while consuming the evidence budget.
            if (SKIP_MID_IN_EXPANSION and is_mid(disc_entity)
                    and disc_entity != target_mid):
                continue

            for rel, tails in KG[disc_entity].items():
                actual_rel = rel.lstrip("~")
                if actual_rel in SENSITIVE_RELATIONS:
                    continue

                for tail in tails:
                    if rel.startswith("~"):
                        triple = [tail, actual_rel, disc_entity]
                    else:
                        triple = [disc_entity, rel, tail]

                    # Drop facts about other MIDs outright: they are unreadable to
                    # the model ("/m/EXAMPLE01 religion Christianity" names nobody) and
                    # crowd out lines that carry an actual name.
                    if (SKIP_MID_IN_EXPANSION and is_mid(tail)
                            and tail != target_mid):
                        continue

                    # Keep triple if it links back into evidence or is human-readable
                    if (tail in discovered_entities
                            or tail == target_mid
                            or not is_mid(tail)):
                        multihop_triples.append(triple)
                        if tail not in visited_expanders:
                            next_frontier.add(tail)

        discovered_entities.update(next_frontier)
        frontier = next_frontier

    all_evidence = confirmed_triples + multihop_triples

    # ─── PHA D: Score + deduplicate + truncate (connectivity-aware) ─────────
    deduplicated = _deduplicate_and_filter(all_evidence, max_triples)

    # Distance of every entity from the target, measured INSIDE the candidate
    # subgraph. Used both to score triples and, at the end, to drop fragments
    # that cannot be reached from the target at all.
    hop_of = _hop_distances(deduplicated, target_mid)

    # ---- Neutral mode: no semantic scoring, order purely by distance ----
    # Hands the model the raw local neighbourhood (all 1-hop facts, then 2-hop,
    # ...) and lets it decide what is informative, instead of baking in our own
    # notion of "useful" via IDF and bonus terms.
    if EVIDENCE_SELECTION == "hop":
        reachable = [trip for trip in deduplicated
                     if trip[0] in hop_of and trip[2] in hop_of]
        reachable.sort(key=lambda tr: max(hop_of[tr[0]], hop_of[tr[2]]))
        return reachable[:max_triples]

    # Score by relation IDF + connects-to-context bonus + selected-relation bonus
    # Context bonus only for MID context entities (anonymized persons like spouses/
    # siblings). String entities like "United Artists Corporation" are already
    # captured as 1-hop facts — giving them a bonus would promote noisy triples
    # connected to the company rather than to the target person.
    mid_context_entities = {e for e in context_entities if is_mid(e)}
    scored = []
    for trip in deduplicated:
        h, r, t = trip[0], trip[1], trip[2]
        score = relation_idf.get(r, 0.0)
        if h in mid_context_entities or t in mid_context_entities:
            score += EVIDENCE_ALPHA
        if h == target_mid or t == target_mid:
            score += EVIDENCE_ALPHA  # triples directly involving target are gold
        if r in selected_set:
            score += EVIDENCE_BETA
        # Discrimination bonus: rare (r, value) pairs get higher priority.
        # Only applied to target-direct triples with a human-readable value.
        if h == target_mid or t == target_mid:
            val = t if h == target_mid else h
            if not is_mid(val):
                sharing = sum(1 for e in KG if r in KG[e] and val in KG[e][r])
                if sharing > 0:
                    score += EVIDENCE_GAMMA / sharing

        # Proximity bonus: a triple the attacker can actually REACH from the
        # target is worth far more than an equally "interesting" fact floating
        # in a disconnected corner. Scoring each triple in isolation used to let
        # a high-IDF edge survive truncation while the edges linking it back to
        # the target were cut, leaving orphan fragments (e.g. the evidence kept
        # "James K. Polk -> Tennessee" but dropped "politician -> James K. Polk",
        # so the name was visible yet structurally unreachable).
        d = min(hop_of.get(h, 99), hop_of.get(t, 99))
        if d <= EVIDENCE_MAX_HOP_KEEP:
            score += EVIDENCE_PROXIMITY / (1.0 + d)
        scored.append((score, trip))

    # Higher score first, but stable-tie-break by original order
    scored.sort(key=lambda x: -x[0])
    sorted_triples = [trip for _, trip in scored]

    final_evidence = _graph_extractor(sorted_triples)
    final_evidence = final_evidence[:max_triples]

    # Keep only what is still connected to the target after truncation, so the
    # model never sees a name it has no structural way of reaching.
    if EVIDENCE_REQUIRE_CONNECTED:
        final_evidence = _keep_connected_component(final_evidence, target_mid)

    return final_evidence


def _hop_distances(triples, target_mid):
    """
    BFS distance from target_mid to every entity, over the candidate subgraph
    only (undirected). Entities in other components simply never get a value,
    so callers treat them as unreachable.
    """
    adj = defaultdict(set)
    for trip in triples:
        h, t = trip[0], trip[2]
        adj[h].add(t)
        adj[t].add(h)

    dist = {target_mid: 0}
    queue = deque([target_mid])
    while queue:
        node = queue.popleft()
        for nxt in adj[node]:
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    return dist


def _keep_connected_component(triples, target_mid):
    """
    Drop triples that end up in a component not containing the target.

    Truncation can sever the edges that linked a fact back to the target,
    leaving orphan fragments: the name is present in the evidence but there is
    no chain of facts from the target to it, so the only way the model can
    "use" it is by recognising it from outside knowledge. Removing those keeps
    the evidence honest — everything shown is something the attacker could
    actually have walked to.
    """
    if not triples:
        return triples

    dist = _hop_distances(triples, target_mid)
    if len(dist) <= 1:          # target isolated — nothing to prune against
        return triples

    return [trip for trip in triples
            if trip[0] in dist and trip[2] in dist]


def _deduplicate_and_filter(triples, max_triples):
    """
    Remove duplicate triples and normalize reverse relations.
    Adapted from KG-GPT lines 451-461.
    """
    seen = set()
    deduplicated = []
    
    for trip in triples:
        h, r, t = trip[0], trip[1], trip[2]
        
        # Normalize: if relation has ~, flip to canonical form
        if r.startswith("~"):
            h, r, t = t, r[1:], h
        
        key = (h, r, t)
        reverse_key = (t, r, h)
        
        if key not in seen and reverse_key not in seen:
            seen.add(key)
            deduplicated.append([h, r, t])
    
    return deduplicated


def _graph_extractor(evidence_list):
    """
    Filter evidence to reduce redundancy.
    Adapted from KG-GPT's graph_extractor() (lines 186-247).
    
    Logic: for each (head, rel) or (tail, rel) pair, keep at most 
    one representative triple to avoid flooding with similar triples.
    """
    if not evidence_list:
        return evidence_list
    
    return_list = []
    filter_dict = {"head": {}, "tail": {}}
    
    # Always keep first triple
    return_list.append(evidence_list[0])
    used_heads = [evidence_list[0][0]]
    used_tails = [evidence_list[0][2]]
    filter_dict["head"][evidence_list[0][0]] = [evidence_list[0][1]]
    filter_dict["tail"][evidence_list[0][2]] = [evidence_list[0][1]]
    
    for trip in evidence_list[1:]:
        if trip in return_list:
            continue
        
        h, r, t = trip[0], trip[1], trip[2]
        
        # Check if this (head, relation) pair already used
        skip = False
        if h in filter_dict["head"] and r in filter_dict["head"][h]:
            skip = True
        if t in filter_dict["tail"] and r in filter_dict["tail"][t]:
            skip = True
        
        if skip:
            continue
        
        # Add triple and update filter
        return_list.append(trip)
        filter_dict["head"].setdefault(h, []).append(r)
        filter_dict["tail"].setdefault(t, []).append(r)
        used_heads.append(h)
        used_tails.append(t)
    
    return return_list


# ============================================================
# Iterative Evidence Expansion
# ============================================================

def expand_evidence_from_feedback(KG, target_mid, existing_evidence,
                                   retrieval_requests, max_new_triples=15,
                                   respect_sensitive=False):
    """
    Mở rộng evidence dựa trên EXPLORE requests từ LLM.

    LLM chỉ được yêu cầu EXPLORE entity ĐÃ XUẤT HIỆN trong existing_evidence.
    Mục tiêu: bổ sung context cho các MID ẩn danh đang là neighbor của target,
    không cho LLM "khai" entity ngoài evidence.

    Args:
        KG: dict, KG[entity][relation] = [tails]
        target_mid: str
        existing_evidence: list of existing [h, r, t] triples
        retrieval_requests: dict with entities list
        max_new_triples: int, max triples to add per round
        respect_sensitive: when False (default), sensitive-relation structural
            triples are included — the attacker legitimately has access to the
            structural anonymized KG (e.g., how many siblings a MID has),
            and showing sibling/spouse network is critical for entity resolution.
            Set True only to reproduce the old conservative behavior.

    Returns:
        expanded_evidence: list of ALL triples (existing + new)
        new_triples_count: int, how many new triples were added
    """
    existing_keys = set()
    entities_in_evidence = set([target_mid])
    for trip in existing_evidence:
        h, r, t = trip[0], trip[1], trip[2]
        existing_keys.add((h, r, t))
        existing_keys.add((t, r, h))
        entities_in_evidence.add(h)
        entities_in_evidence.add(t)

    targeted_triples = []   # from NEED_RELATION — highest priority, model asked precisely
    broad_triples = []      # from EXPLORE — whole-node dump
    skipped_oov = []
    skipped_mid = []        # requests aimed at another MID (dead end, see config)

    def _emit(entity, rel, tail, bucket):
        actual_rel = rel.lstrip("~")
        if respect_sensitive and actual_rel in SENSITIVE_RELATIONS:
            return
        # Facts about a different MID name nobody — don't spend the round's budget
        # on them (the target's own facts stay, they are the problem statement).
        if SKIP_MID_IN_EXPANSION and is_mid(tail) and tail != target_mid:
            return
        triple = [tail, actual_rel, entity] if rel.startswith("~") else [entity, rel, tail]
        key = (triple[0], triple[1], triple[2])
        if key not in existing_keys:
            existing_keys.add(key)
            bucket.append(triple)

    # ─── 1. Targeted pulls: NEED_RELATION <relation> of <node> ───────────
    # The model named BOTH the node and the kind of fact it wants. Serve these
    # first so precise reasoning is actually rewarded with the right material.
    for entity, rel_sub in retrieval_requests.get("relation_reqs", []):
        entity = (entity or "").strip()
        if not entity or entity not in entities_in_evidence:
            if entity:
                skipped_oov.append(f"{entity} (NEED_RELATION)")
            continue
        if SKIP_MID_IN_EXPANSION and is_mid(entity) and entity != target_mid:
            skipped_mid.append(f"{entity} (NEED_RELATION)")
            continue
        if entity not in KG:
            continue
        for rel, tails in KG[entity].items():
            if rel_sub.lower() not in rel.lstrip("~").lower():
                continue
            for tail in tails:
                _emit(entity, rel, tail, targeted_triples)

    # ─── 2. Broad pulls: EXPLORE / IDENTIFY <node> ──────────────────────
    for entity in retrieval_requests.get("entities", []):
        entity = entity.strip()
        if not entity:
            continue
        if entity not in entities_in_evidence:
            skipped_oov.append(entity)
            continue
        if SKIP_MID_IN_EXPANSION and is_mid(entity) and entity != target_mid:
            skipped_mid.append(entity)
            continue
        if entity not in KG:
            continue
        for rel, tails in KG[entity].items():
            for tail in tails:
                _emit(entity, rel, tail, broad_triples)

    # Targeted results keep their full budget; broad results fill what's left.
    new_triples = targeted_triples[:max_new_triples]
    remaining = max_new_triples - len(new_triples)
    if remaining > 0:
        new_triples += broad_triples[:remaining]

    expanded_evidence = existing_evidence + new_triples

    if skipped_oov:
        # Surface OOV requests so the orchestrator can flag confirmation-bias attempts
        print(f"    ⚠️  Request rejected (entity not in current evidence): {skipped_oov}")
    if skipped_mid:
        print(f"    ⏭️  Request skipped (anonymized MID — cannot be named): {skipped_mid}")

    return expanded_evidence, len(new_triples)

