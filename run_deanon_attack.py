"""
KG De-Anonymization Attack Pipeline
====================================
Adapted from KG-GPT (Fact Verification) for Identity Inference.

Pipeline:
  Step 1: Target Identification  — Scan sensitive triples → target MIDs
  Step 2: Relation Retrieval     — KG[mid] → candidates → LLM top-K
  Step 3: Evidence Construction  — Query patterns → verify → multi-hop join
  Step 4: LLM Identity Inference — Verbalize → prompt → predict real name

Usage:
  # Dry run (no LLM calls, show targets + evidence):
  python run_deanon_attack.py --dry_run --n_samples 10

  # Full attack:
  python run_deanon_attack.py --n_samples 10

  # With explicit API key:
  python run_deanon_attack.py --api_key nvapi-xxx --n_samples 10
"""

import sys
import os
import json
import time

# Force UTF-8 output so Unicode in evidence/entity names doesn't crash on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import argparse
from collections import defaultdict

from deanon_pipeline.config import (
    ANON_DATA_DIR, WIKI_MAPPING_PATH, RESULTS_DIR,
    SENSITIVE_RELATIONS, TOP_K_RELATIONS, N_HOPS, MAX_EVIDENCE_TRIPLES,
    NVIDIA_API_KEY, N_REFINEMENT_ROUNDS,
    MAX_NEW_TRIPLES_PER_ROUND, MAX_EXPLORE_PER_ROUND,
)
from deanon_pipeline.data_loader import (
    load_all_train_triples, build_kg_dict, load_wiki_mapping, compute_kg_stats,
)
from deanon_pipeline.step1_target_identification import identify_targets, select_targets
from deanon_pipeline.step2_relation_retrieval import (
    retrieve_relations, clean_relation_for_display,
    retrieve_relations_for_placeholder,
)
from deanon_pipeline.step3_evidence_construction import construct_evidence_graph, expand_evidence_from_feedback
from deanon_pipeline.step4_llm_inference import (
    infer_identity, infer_with_feedback, verbalize_evidence,
    build_placeholder_map,
)
from deanon_pipeline.evaluation import check_match, compute_metrics, print_metrics


# Write a .partial.json every N targets so an interrupted run keeps its
# reasoning traces. 0 disables checkpointing.
CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", "25"))

# These sleeps were sized for the Google AI Studio free tier (20 req/min). The
# xah.io proxy has no such cap — 4 concurrent calls returned in 12.5s and a
# 25-target run produced zero 429s — so they now default to 0. They cost 6s per
# target plus 4s per round (~20s/target at the observed 3.5 rounds), i.e. about
# an hour per 210-target experiment spent doing nothing.
# Set INTER_TARGET_SLEEP / INTER_ROUND_SLEEP if a future backend does rate-limit.
INTER_TARGET_SLEEP = float(os.getenv("INTER_TARGET_SLEEP", "0"))
INTER_ROUND_SLEEP = float(os.getenv("INTER_ROUND_SLEEP", "0"))

# Reuse targets already completed in a previous run's .partial.json.
# Set RESUME=0 to force a clean re-run.
RESUME = os.getenv("RESUME", "1") not in ("0", "false", "False")


def _resolve_out(out_name):
    """Resolve --output to an absolute path under RESULTS_DIR.

    --output may be a bare filename, an absolute path, or a path that already
    carries the results-dir prefix ("deanon_results/x.json"); joining blindly
    turned the last form into deanon_results/deanon_results/x.json.
    """
    if os.path.isabs(out_name):
        return out_name
    rel = out_name.replace("\\", "/").lstrip("./")
    prefix = os.path.basename(RESULTS_DIR) + "/"
    while rel.startswith(prefix):
        rel = rel[len(prefix):]
    return os.path.normpath(os.path.join(RESULTS_DIR, rel))


def main():
    parser = argparse.ArgumentParser(
        description="KG De-Anonymization Attack Pipeline (KG-GPT Adapted)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="NVIDIA API Key (or set NVIDIA_API_KEY env var)")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of targets to attack")
    parser.add_argument("--top_k", type=int, default=TOP_K_RELATIONS,
                        help="Top-K relations for LLM selection")
    parser.add_argument("--n_hops", type=int, default=N_HOPS,
                        help="Multi-hop expansion depth")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show targets + evidence only, skip LLM calls")
    parser.add_argument("--mode", type=str, default="open_book",
                        choices=["open_book", "closed_book"],
                        help="open_book: LLM + world knowledge | closed_book: evidence only")
    parser.add_argument("--target_relations", type=str, default=None,
                        help="Comma-separated relations to restrict target selection. "
                             "Default: no restriction (attack all targets with ground truth).")
    parser.add_argument("--fuzzy_match", action="store_true",
                        help="Allow partial name match (substring / all-parts). "
                             "Default: exact normalized match only.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename (relative to RESULTS_DIR). "
                             "Default: attack_results.json")
    parser.add_argument("--target_mids_file", type=str, default=None,
                        help="Path to JSON file (list of {mid, real_name} dicts) produced by "
                             "create_test_set.py. When set, attacks ONLY those exact MIDs in "
                             "that order — enables fair comparison across approaches.")
    args = parser.parse_args()

    target_relations_filter = None
    if args.target_relations:
        target_relations_filter = [r.strip() for r in args.target_relations.split(",") if r.strip()]

    api_key = args.api_key or NVIDIA_API_KEY

    if not api_key and not args.dry_run:
        print("ERROR: Set NVIDIA_API_KEY env var or pass --api_key")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ═══════════════════════════════════════════════════════
    # LOAD DATA
    # ═══════════════════════════════════════════════════════
    print("=" * 65)
    print("  KG DE-ANONYMIZATION ATTACK PIPELINE")
    print("  Adapted from KG-GPT for Identity Inference")
    mode_label = "OPEN-BOOK (World Knowledge + Evidence)" if args.mode == "open_book" else "CLOSED-BOOK (Evidence Only)"
    print(f"  Mode: {mode_label}")
    print("=" * 65)

    print("\n[LOAD] Loading data...")
    triples = load_all_train_triples(ANON_DATA_DIR)
    print(f"  Loaded {len(triples):,} train triples")

    print("[LOAD] Building KG dict (KG-GPT compatible)...")
    KG = build_kg_dict(triples)
    print(f"  KG index: {len(KG):,} unique entities")

    print("[LOAD] Computing KG statistics (degree, IDF, person filter)...")
    kg_stats = compute_kg_stats(triples)
    print(f"  Hub threshold (95th pct): {kg_stats['hub_threshold']:.1f}")
    print(f"  Person-likely entities: {len(kg_stats['person_entities']):,}")

    print("[LOAD] Loading wiki mapping (ground truth)...")
    wiki_mapping = load_wiki_mapping(WIKI_MAPPING_PATH)
    print(f"  Loaded {len(wiki_mapping):,} MID -> name mappings")

    # A dataset may relabel its entities — the pseudonymised variant replaces every
    # person with PERSON_0001-style codes so no real name can be recognised. When it
    # does, victims.json is the authority on what the answer looks like IN THAT GRAPH,
    # and scoring has to use it; wiki_mapping still holds the underlying real names.
    dataset_labels = {}
    _victims_path = os.path.join(ANON_DATA_DIR, "victims.json")
    if os.path.exists(_victims_path):
        with open(_victims_path, encoding="utf-8") as _f:
            dataset_labels = json.load(_f)
        _relabelled = sum(1 for m, n in dataset_labels.items()
                          if wiki_mapping.get(m) and wiki_mapping[m] != n)
        if _relabelled:
            print(f"  Dataset uses its own labels for {_relabelled:,} victims "
                  f"(e.g. {next(iter(dataset_labels.values()))}) — scoring against those")

    # ═══════════════════════════════════════════════════════
    # STEP 1: TARGET IDENTIFICATION
    # ═══════════════════════════════════════════════════════
    print(f"\n{'-'*65}")
    print(f"  STEP 1: Target Identification")
    print(f"{'-'*65}")

    all_targets = identify_targets(triples, SENSITIVE_RELATIONS)
    print(f"  Found {len(all_targets)} unique anonymized entities")

    # Distribution
    rel_count = defaultdict(int)
    for mid, info in all_targets.items():
        for ctx in info["sensitive_contexts"]:
            rel_count[ctx["relation"]] += 1
    print("\n  Sensitive relation distribution:")
    for rel, count in sorted(rel_count.items(), key=lambda x: -x[1]):
        print(f"    {count:5d}x  {rel}")

    # Mapping coverage diagnostic
    n_with_gt = sum(1 for mid in all_targets if wiki_mapping.get(mid))
    coverage = n_with_gt / len(all_targets) if all_targets else 0
    print(f"\n  Mapping coverage: {n_with_gt}/{len(all_targets)} ({coverage:.1%}) "
          f"targets have ground truth")
    if target_relations_filter:
        print(f"  Target-relation filter: {target_relations_filter}")

    # Select top targets
    if args.target_mids_file:
        import json as _json
        with open(args.target_mids_file, encoding="utf-8") as _f:
            _fixed = _json.load(_f)
        _fixed_mids = [entry["mid"] for entry in _fixed]
        # Rebuild from all_targets preserving fixed order
        selected = []
        for _mid in _fixed_mids:
            if _mid not in all_targets:
                print(f"  [WARN] {_mid} not in all_targets — skipping")
                continue
            _info = all_targets[_mid]
            # Ground truth must be spelled the way the GRAPH spells it. On a
            # pseudonymised dataset the graph says "PERSON_0465", so scoring against
            # wiki_mapping's "Harvey Weinstein" would mark every correct answer wrong.
            # dataset_labels comes from that dataset's victims.json; wiki_mapping is
            # the fallback for datasets that carry real names.
            _real = dataset_labels.get(_mid) or wiki_mapping.get(_mid)
            if not _real:
                print(f"  [WARN] {_mid} has no ground truth — skipping")
                continue
            import re as _re2
            _all_rels = list(KG.get(_mid, {}).keys())
            _pub = [r for r in _all_rels
                    if r not in SENSITIVE_RELATIONS and r.lstrip("~") not in SENSITIVE_RELATIONS]
            selected.append({
                "mid": _mid,
                "real_name": _real,
                "n_public_relations": len(_pub),
                "n_sensitive": len(_info["sensitive_contexts"]),
                "sensitive_contexts": _info["sensitive_contexts"],
            })
        print(f"  [FIXED TEST SET] Loaded {len(selected)} targets from {args.target_mids_file}")
    else:
        # Same reason as above: on a relabelled dataset the graph's own names are
        # the ones an answer has to match, so selection scores against them too.
        selected = select_targets(
            all_targets, KG, {**wiki_mapping, **dataset_labels},
            SENSITIVE_RELATIONS, args.n_samples,
            target_relations=target_relations_filter,
        )
    print(f"\n  Selected {len(selected)} targets for attack:\n")
    print(f"  {'#':>3}  {'MID':<15} {'Ground Truth':<30} {'Public Rels':>11}")
    print(f"  {'-'*62}")
    for i, t in enumerate(selected, 1):
        print(f"  {i:>3}  {t['mid']:<15} {t['real_name']:<30} "
              f"{t['n_public_relations']:>11}")

    # ═══════════════════════════════════════════════════════
    # STEP 2→4: ATTACK LOOP
    # ═══════════════════════════════════════════════════════
    # Resume: reuse targets already finished in a previous (interrupted) run.
    # A long run can be cut short by sleep/standby or by the proxy withdrawing a
    # model mid-experiment; without this the whole thing restarts from zero.
    results = []
    if RESUME and args.output:
        _pk = _resolve_out(args.output)
        _pk = _pk[:-5] + ".partial.json" if _pk.endswith(".json") else _pk + ".partial"
        if os.path.exists(_pk):
            try:
                with open(_pk, encoding="utf-8") as _f:
                    _prev = json.load(_f)
                _done = _prev.get("results", _prev) if isinstance(_prev, dict) else _prev
                _by_mid = {r["mid"]: r for r in _done if r.get("mid")}
                _keep = [t for t in selected if t["mid"] in _by_mid]
                if _keep:
                    results = [_by_mid[t["mid"]] for t in _keep]
                    _skip = {t["mid"] for t in _keep}
                    selected = [t for t in selected if t["mid"] not in _skip]
                    print(f"\n  [RESUME] {len(results)} target đã xong từ {os.path.basename(_pk)}"
                          f" — còn {len(selected)} target")
            except Exception as _e:
                print(f"  [RESUME failed, chạy lại từ đầu] {_e}")

    _offset = len(results)
    _total = _offset + len(selected)      # tổng thật, dùng cho hiển thị + checkpoint
    for i, target in enumerate(selected, 1):
        i += _offset          # keep checkpoint counter aligned with total done
        mid = target["mid"]
        real_name = target["real_name"]

        if i > 1 and not args.dry_run and INTER_TARGET_SLEEP:
            time.sleep(INTER_TARGET_SLEEP)

        print(f"\n{'='*65}")
        print(f"  TARGET {i}/{_total}:  {mid}")
        print(f"  Ground Truth:      {real_name}")
        print(f"{'='*65}")

        # ── STEP 2: Relation Retrieval ──
        print(f"\n  > Step 2: Relation Retrieval")

        if args.dry_run:
            # Dry run: use all non-sensitive relations (no LLM call)
            from deanon_pipeline.step2_relation_retrieval import get_candidate_relations
            candidates = get_candidate_relations(
                KG, mid, target["sensitive_contexts"], kg_stats=kg_stats,
            )
            selected_rels = candidates[:args.top_k]
            print(f"    [DRY RUN] Using first {len(selected_rels)} candidates")
        else:
            selected_rels, candidates = retrieve_relations(
                KG, mid, target["sensitive_contexts"], top_k=args.top_k,
                api_key=api_key, kg_stats=kg_stats,
            )
            print(f"    Candidates: {len(candidates)} -> LLM selected: {len(selected_rels)}")

        print(f"    Selected relations:")
        for rel in selected_rels[:8]:
            print(f"      - {clean_relation_for_display(rel)}  ({rel})")

        # ── STEP 3: Evidence Construction ──
        print(f"\n  > Step 3: Evidence Graph Construction")
        evidence = construct_evidence_graph(
            KG, mid, target["sensitive_contexts"], selected_rels,
            n_hops=args.n_hops, max_triples=MAX_EVIDENCE_TRIPLES,
            kg_stats=kg_stats,
        )
        print(f"    Evidence: {len(evidence)} triples")

        # Show evidence (with typed placeholders for masked neighbors)
        placeholder_map = build_placeholder_map(mid, evidence, target["sensitive_contexts"])
        sentences = verbalize_evidence(mid, evidence, wiki_mapping=wiki_mapping,
                                       placeholder_map=placeholder_map)
        if placeholder_map:
            print(f"    Masked neighbors -> placeholders: "
                  f"{', '.join(f'{m}->{lbl}' for m, lbl in placeholder_map.items())}")
        print(f"    Top evidence facts:")
        for j, s in enumerate(sentences[:10], 1):
            print(f"      {j:>2}. {s}")
        if len(sentences) > 10:
            print(f"      ... +{len(sentences)-10} more")

        # Show sensitive context
        print(f"\n    Sensitive context:")
        for ctx in target["sensitive_contexts"][:5]:
            ctx_name = ctx["context_entity"]
            print(f"      [S] {ctx['relation'].split('/')[-1]}: {ctx_name}")

        # ── STEP 4: Iterative LLM Inference (N rounds) ──
        print(f"\n  > Step 4: Iterative LLM Inference ({N_REFINEMENT_ROUNDS} rounds)")

        if args.dry_run:
            predictions = []
            raw_response = "[DRY RUN - no LLM call]"
            print(f"    [DRY RUN] Skipping LLM call")

            save_path = os.path.join(RESULTS_DIR, f"evidence_{mid.replace('/', '_')}.json")
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "target_mid": mid,
                    "real_name": real_name,
                    "evidence": evidence,
                    "evidence_sentences": sentences,
                    "sensitive_contexts": target["sensitive_contexts"],
                }, f, ensure_ascii=False, indent=2)
            all_round_responses = []
        else:
            current_evidence = evidence
            previous_predictions = None
            previous_placeholder_identities = None
            placeholder_identity_history = {}  # mid -> [(round, identity, conf)]
            all_round_responses = []
            barren_rounds = 0   # consecutive rounds that yielded no new evidence
            failed_requests = []  # requests already tried that returned nothing
            conversation = []     # multi-turn history so the model remembers its own plan
            newly_added = []      # triples pulled in since the previous round ([NEW] lines)

            for rnd in range(1, N_REFINEMENT_ROUNDS + 1):
                print(f"\n    [ Round {rnd}/{N_REFINEMENT_ROUNDS} "
                      f"| evidence: {len(current_evidence)} triples ]")

                predictions, retrieval_reqs, raw_response, prompt, top1_conf, ph_identities = (
                    infer_with_feedback(
                        mid, current_evidence, target["sensitive_contexts"],
                        round_num=rnd, max_rounds=N_REFINEMENT_ROUNDS,
                        previous_predictions=previous_predictions,
                        api_key=api_key, mode=args.mode, KG=KG,
                        candidates=None,  # open-vocab only (closed-set narrowing removed)
                        wiki_mapping=wiki_mapping,
                        previous_placeholder_identities=previous_placeholder_identities,
                        kg_stats=kg_stats,
                        failed_requests=failed_requests,
                        conversation=conversation,
                        new_triples=newly_added,
                    )
                )

                # Track placeholder identities
                for ph_mid, info in (ph_identities or {}).items():
                    placeholder_identity_history.setdefault(ph_mid, []).append(
                        (rnd, info.get("identity", "UNKNOWN"), info.get("confidence", 0.0))
                    )
                if ph_identities:
                    print(f"    [PH] Placeholder hypotheses:")
                    for ph_mid, info in ph_identities.items():
                        print(f"       {ph_mid} -> {info['identity']} ({info['confidence']:.0f}%)")
                
                # Display response (filtered)
                import re as _re
                display = _re.sub(r'<think>.*?</think>', '', raw_response, flags=_re.DOTALL).strip()
                if not display:
                    display = raw_response
                print(f"    [LLM] Round {rnd} Response:")
                for line in display.split('\n'):
                    print(f"       {line}")
                
                # ── Split the model's reasoning into its two analysable parts ──
                # (a) identification reasoning: PLACEHOLDER_IDENTITIES + PREDICTIONS
                # (b) retrieval reasoning: the RETRIEVAL_REQUEST rationales — i.e.
                #     WHY it decided those facts were worth fetching.
                import re as _re2
                _rr_split = _re2.split(r'RETRIEVAL_REQUEST\s*:', raw_response,
                                       maxsplit=1, flags=_re2.IGNORECASE)
                identification_reasoning = _rr_split[0].strip()
                retrieval_reasoning = _rr_split[1].strip() if len(_rr_split) > 1 else ""

                all_round_responses.append({
                    "round": rnd,
                    "predictions": predictions,
                    "retrieval_requests": retrieval_reqs,
                    "raw_response": raw_response,
                    # --- reasoning traces for later analysis ---
                    "identification_reasoning": identification_reasoning,
                    "retrieval_reasoning": retrieval_reasoning,
                    "evidence_at_round_start": [list(t) for t in current_evidence],
                    # Facts that arrived because of the PREVIOUS round's request —
                    # i.e. what this round actually got to reason over as [NEW].
                    "new_triples_received": [list(t) for t in newly_added],
                    "n_evidence": len(current_evidence),
                    "n_new_triples_received": len(newly_added),
                    "top1_confidence": top1_conf,
                    "placeholder_identities": ph_identities or {},
                    "failed_requests_so_far": list(failed_requests),
                })

                # Early-stop: high-confidence convergence on the same top-1
                if (rnd >= 2 and top1_conf is not None and top1_conf >= 85
                        and previous_predictions and predictions
                        and previous_predictions[0].lower() == predictions[0].lower()):
                    print(f"    [OK] Converged: top-1 stable @ {top1_conf:.0f}% - stopping early")
                    break

                # Confidence-gated previous_predictions for next round
                # (Phase 5 fix: low-confidence predictions are no longer fed
                # forward — that was the source of the Michael Jackson →
                # "Jam Master Jay" confirmation bias.)
                CONFIDENCE_GATE = 70.0
                if top1_conf is not None and top1_conf >= CONFIDENCE_GATE:
                    previous_predictions = predictions
                else:
                    previous_predictions = None  # don't anchor LLM on uncertain guess
                    if top1_conf is not None:
                        print(f"    [!] Top-1 confidence {top1_conf:.0f}% < {CONFIDENCE_GATE:.0f}% - "
                              f"NOT feeding predictions forward (avoid confirmation bias)")

                # Confidence-gated placeholder hypotheses (lower bar than TARGET — these
                # are sub-tasks, an unsure placeholder hint is still useful context).
                PH_CONFIDENCE_GATE = 60.0
                forwarded_ph = {}
                if ph_identities:
                    for ph_mid, info in ph_identities.items():
                        if info.get("identity", "UNKNOWN").upper() == "UNKNOWN":
                            continue
                        if info.get("confidence", 0.0) >= PH_CONFIDENCE_GATE:
                            forwarded_ph[ph_mid] = info
                previous_placeholder_identities = forwarded_ph if forwarded_ph else None

                # Nếu chưa phải round cuối → mở rộng evidence
                if rnd < N_REFINEMENT_ROUNDS:
                    rel_reqs = retrieval_reqs.get("relation_reqs", [])
                    has_requests = bool(retrieval_reqs["entities"]) or bool(rel_reqs)

                    if has_requests:
                        if rel_reqs:
                            print(f"\n    [NEED_RELATION] Targeted pulls:")
                            for _n, _r in rel_reqs:
                                print(f"       - {_r}  of  {_n}")
                        if retrieval_reqs["entities"]:
                            print(f"    [EXPLORE/IDENTIFY] Expanding nodes:")
                            for ent in retrieval_reqs["entities"]:
                                print(f"       - {ent}")

                        # Step 1: standard expansion via expand_evidence_from_feedback
                        _before_len = len(current_evidence)
                        current_evidence, n_new = expand_evidence_from_feedback(
                            KG, mid, current_evidence, retrieval_reqs,
                            max_new_triples=MAX_NEW_TRIPLES_PER_ROUND,
                        )
                        newly_added = current_evidence[_before_len:]  # for [NEW] markers
                        print(f"    [+] Standard expansion: +{n_new} new triples")

                        # Step 2 (Component D): per-placeholder targeted Step 2 retrieval
                        # For each placeholder the LLM requested, run a focused Step 2
                        # to find this placeholder's most discriminative relations, then
                        # add evidence triples from those relations.
                        target_facts = [
                            f"{ctx['relation'].split('/')[-1]} -> {ctx['context_entity']}"
                            for ctx in target["sensitive_contexts"][:5]
                        ]
                        extra_count = 0
                        for ph_mid in retrieval_reqs["entities"][:MAX_EXPLORE_PER_ROUND]:
                            if ph_mid == mid or ph_mid not in KG:
                                continue
                            try:
                                ph_rels, _ = retrieve_relations_for_placeholder(
                                    KG, ph_mid, target_facts,
                                    kg_stats=kg_stats, top_k=5, api_key=api_key,
                                )
                            except Exception as e:
                                print(f"       [PH-Step2 error] {ph_mid}: {e}")
                                continue
                            if not ph_rels:
                                continue
                            ph_ev = construct_evidence_graph(
                                KG, ph_mid, [], ph_rels,
                                n_hops=1, max_triples=10, kg_stats=kg_stats,
                            )
                            existing_keys = {tuple(t) for t in current_evidence}
                            new_ph_ev = [t for t in ph_ev if tuple(t) not in existing_keys]
                            current_evidence.extend(new_ph_ev)
                            newly_added = list(newly_added) + new_ph_ev
                            extra_count += len(new_ph_ev)
                        if extra_count > 0:
                            print(f"    [++] Per-placeholder Step 2: +{extra_count} more triples "
                                  f"(total: {len(current_evidence)})")

                        if n_new == 0 and extra_count == 0:
                            # Remember these dead ends so the next prompt tells the
                            # model not to repeat them.
                            for _n, _r in rel_reqs:
                                _msg = f"NEED_RELATION: {_r} of {_n}"
                                if _msg not in failed_requests:
                                    failed_requests.append(_msg)
                            for _e in retrieval_reqs["entities"]:
                                _msg = f"EXPLORE: {_e}"
                                if _msg not in failed_requests:
                                    failed_requests.append(_msg)
                            barren_rounds += 1
                            if barren_rounds >= 2:
                                print(f"    [STOP] Two consecutive rounds returned no new "
                                      f"evidence. Stopping.")
                                break
                            print(f"    [!] Request returned nothing new — letting the model "
                                  f"try a different node next round.")
                        else:
                            barren_rounds = 0
                    else:
                        newly_added = []
                        barren_rounds += 1
                        if barren_rounds >= 2:
                            print(f"\n    [STOP] No retrieval requests for two rounds. Stopping.")
                            break
                        print(f"\n    [!] No retrieval requests this round — continuing.")

                    if INTER_ROUND_SLEEP:
                        time.sleep(INTER_ROUND_SLEEP)

                print(f"    {'-'*52}")

        # ── Evaluate (dùng predictions của round cuối cùng) ──
        eval_result = check_match(predictions, real_name, strict=not args.fuzzy_match)
        tag = "[HIT]" if eval_result["match"] else "[MISS]"
        rank_info = f" (rank {eval_result['rank']})" if eval_result["match"] else ""
        print(f"\n    Result: {tag}{rank_info}  |  Ground Truth: {real_name}")

        results.append({
            "mid": mid,
            "real_name": real_name,
            "n_evidence": len(evidence),
            "n_evidence_final": len(current_evidence) if not args.dry_run else len(evidence),
            "predictions": predictions,
            "rounds": all_round_responses,
            # Full multi-turn transcript: every user turn (task + [NEW] deltas) and
            # every assistant turn (its reasoning), in order — for later analysis of
            # how the model reasoned and how it steered its own retrieval.
            "conversation": conversation if not args.dry_run else [],
            "rounds_used": len(all_round_responses),
            "failed_requests": failed_requests if not args.dry_run else [],
            "match": eval_result["match"],
            "rank": eval_result["rank"],
            "sensitive_contexts": target["sensitive_contexts"],
        })

        # ── Checkpoint ─────────────────────────────────────────────────
        # The final JSON is only written after ALL targets finish, so an
        # interrupted run (the proxy has withdrawn a model mid-experiment
        # three times) used to lose every reasoning trace collected so far.
        # Dump partial results periodically to a .partial.json alongside.
        if CHECKPOINT_EVERY and (i % CHECKPOINT_EVERY == 0 or i == _total):
            try:
                _ck = _resolve_out(args.output or "attack_results.json")
                _ck = _ck[:-5] + ".partial.json" if _ck.endswith(".json") else _ck + ".partial"
                os.makedirs(os.path.dirname(_ck) or ".", exist_ok=True)
                with open(_ck, "w", encoding="utf-8") as _f:
                    json.dump({"partial": True, "n_done": len(results),
                               "n_total": _total, "results": results},
                              _f, ensure_ascii=False)
                print(f"    [checkpoint] {len(results)}/{_total} -> {os.path.basename(_ck)}")
            except Exception as _e:
                print(f"    [checkpoint failed] {_e}")

    # ═══════════════════════════════════════════════════════
    # EVALUATION SUMMARY
    # ═══════════════════════════════════════════════════════
    # Guard against a MID appearing twice: resume already de-duplicates by MID,
    # but a metric computed over duplicates would silently be wrong.
    _seen, _uniq = set(), []
    for _r in results:
        if _r.get("mid") in _seen:
            continue
        _seen.add(_r.get("mid"))
        _uniq.append(_r)
    if len(_uniq) != len(results):
        print(f"  [dedup] {len(results)} -> {len(_uniq)} bản ghi (loại {len(results)-len(_uniq)} trùng MID)")
        results = _uniq

    metrics = compute_metrics(results)
    print_metrics(metrics)

    # Save results
    results_path = _resolve_out(args.output if args.output else "attack_results.json")
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {
                "n_samples": args.n_samples,
                "top_k": args.top_k,
                "n_hops": args.n_hops,
                "dry_run": args.dry_run,
                "mode": args.mode,
            },
            "metrics": metrics,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved -> {results_path}")


if __name__ == "__main__":
    main()
