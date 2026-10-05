"""
Non-LLM Baselines for Knowledge Graph Re-identification Attack.
================================================================
Implements classical network de-anonymization and record linkage baselines:
  1. Global Random & 2-hop Candidate Random
  2. Common Neighbors (Shared Anchors count)
  3. Jaccard Similarity
  4. Adamic-Adar Index (Rarity-weighted shared anchors)
  5. Resource Allocation Index (RA / 2-Hop Random Walk probability)
  6. Relation-Aware Path Matching (Predicate-specific IDF weighting)
  7. Personalized PageRank (PPR / Random Walk with Restart via sparse CSR)

Evaluates on all 2,836 victims for FB15k-237 at 5%, 10%, 15% error rates.
Outputs Top-1, Top-3, Top-5 accuracy, MRR, and latency.

Usage:
  python baselines/run_non_llm_baselines.py --rate 05
  python baselines/run_non_llm_baselines.py --all_rates
  python baselines/run_non_llm_baselines.py --test --n_samples 50
"""

import argparse
import collections
import io
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import scipy.sparse as sp

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dataset(rate_str: str) -> Tuple[List[Tuple[str, str, str]], Dict[str, str]]:
    """Load train triples and victims.json for a given move rate.

    '00' is the reference corpus: masked, with no edge misfiled. It is the
    condition a correct labelling produces, so it lives in a differently named
    directory than the leaked corpora.
    """
    if rate_str == "00":
        data_dir = os.path.join(BASE_DIR, "data", "FB15k-237-id-masked")
    else:
        data_dir = os.path.join(BASE_DIR, "data", f"FB15k-237-id-move{rate_str}")
    train_path = os.path.join(data_dir, "train.txt")
    victims_path = os.path.join(data_dir, "victims.json")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train path not found: {train_path}")
    if not os.path.exists(victims_path):
        raise FileNotFoundError(f"Victims path not found: {victims_path}")

    triples = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))

    with open(victims_path, "r", encoding="utf-8") as f:
        victims = json.load(f)

    return triples, victims


def build_graph_structures(triples: List[Tuple[str, str, str]]):
    """
    Build indexing structures for fast graph algorithms:
    - neighbors: node -> set of adjacent nodes
    - incident_edges: node -> list of (rel, neighbor, direction)
    - degree: node -> degree
    - rel_freq: rel -> count
    - all_ind_nodes: set of all IND_XXXX nodes in the graph
    - Sparse transition matrix P_T for Personalized PageRank
    """
    neighbors = collections.defaultdict(set)
    incident_edges = collections.defaultdict(list)
    rel_freq = collections.defaultdict(int)
    all_ind_nodes = set()
    all_nodes_set = set()

    for h, r, t in triples:
        neighbors[h].add(t)
        neighbors[t].add(h)
        incident_edges[h].append((r, t, "out"))
        incident_edges[t].append((r, h, "in"))
        rel_freq[r] += 1
        all_nodes_set.add(h)
        all_nodes_set.add(t)
        if h.startswith("IND_"):
            all_ind_nodes.add(h)
        if t.startswith("IND_"):
            all_ind_nodes.add(t)

    degree = {k: len(v) for k, v in neighbors.items()}
    total_triples = len(triples)
    rel_idf = {r: math.log((total_triples + 1.0) / (cnt + 1.0)) for r, cnt in rel_freq.items()}

    # Adamic-Adar weight per node: 1 / log(max(2, deg))
    aa_weights = {node: 1.0 / math.log(max(2, deg)) for node, deg in degree.items()}
    # Resource allocation weight per node: 1 / deg
    ra_weights = {node: 1.0 / max(1, deg) for node, deg in degree.items()}

    # Index for fast relation-path match: (rel, neighbor, direction) -> list of IND nodes
    rel_path_index = collections.defaultdict(list)
    for ind in all_ind_nodes:
        for r, neigh, direction in incident_edges[ind]:
            rel_path_index[(r, neigh, direction)].append(ind)

    # Index for 2-hop lookup: neighbor -> list of IND nodes
    neighbor_to_ind = collections.defaultdict(list)
    for ind in all_ind_nodes:
        for neigh in neighbors[ind]:
            neighbor_to_ind[neigh].append(ind)

    # Precompute Sparse CSR Matrix for ultra-fast vectorized Personalized PageRank
    all_nodes_list = sorted(all_nodes_set)
    node_to_id = {node: i for i, node in enumerate(all_nodes_list)}
    n_nodes = len(all_nodes_list)

    row_ind = []
    col_ind = []
    data = []
    for u, neighs in neighbors.items():
        u_idx = node_to_id[u]
        deg = len(neighs)
        if deg > 0:
            weight = 1.0 / deg
            for v in neighs:
                row_ind.append(node_to_id[v])  # P^T: transition from u to v means col=u, row=v
                col_ind.append(u_idx)
                data.append(weight)

    P_T = sp.csr_matrix((data, (row_ind, col_ind)), shape=(n_nodes, n_nodes), dtype=np.float32)

    # Array of indices of all IND nodes
    ind_indices = np.array([node_to_id[ind] for ind in sorted(all_ind_nodes)], dtype=np.int32)
    ind_nodes_sorted = sorted(all_ind_nodes)

    return {
        "neighbors": neighbors,
        "incident_edges": incident_edges,
        "degree": degree,
        "rel_idf": rel_idf,
        "aa_weights": aa_weights,
        "ra_weights": ra_weights,
        "all_ind_nodes": ind_nodes_sorted,
        "rel_path_index": rel_path_index,
        "neighbor_to_ind": neighbor_to_ind,
        "node_to_id": node_to_id,
        "P_T": P_T,
        "ind_indices": ind_indices,
    }


def compute_ppr_vector(P_T: sp.csr_matrix, target_idx: int, n_nodes: int, alpha: float = 0.85, max_iter: int = 15) -> np.ndarray:
    """Ultra-fast vectorized PageRank on full graph using sparse matrix multiplication."""
    p = np.zeros(n_nodes, dtype=np.float32)
    p[target_idx] = 1.0
    restart = (1.0 - alpha)
    
    for _ in range(max_iter):
        p = alpha * (P_T.dot(p))
        p[target_idx] += restart

    return p


def evaluate_ranking(scores: Dict[str, float], true_target: str, all_ind_nodes: List[str]) -> Dict[str, float]:
    """
    Given candidate scores and true_target:
    Computes strict rank, average rank (random tie break expectation),
    and whether it is in Top-1, Top-3, Top-5.
    """
    target_score = scores.get(true_target, 0.0)
    
    # Count strictly greater and equal scores among all candidates
    greater_count = 0
    equal_count = 0

    scored_cands = set(scores.keys())
    for cand, score in scores.items():
        if cand == true_target:
            continue
        if score > target_score:
            greater_count += 1
        elif score == target_score:
            equal_count += 1

    # Candidates with score == 0 not in scores dict
    zero_score_cands_count = len(all_ind_nodes) - len(scored_cands)
    if target_score == 0.0:
        equal_count += (zero_score_cands_count - (0 if true_target in scored_cands else 1))
    elif target_score < 0.0:
        greater_count += zero_score_cands_count

    avg_rank = 1.0 + greater_count + 0.5 * equal_count
    min_rank = 1 + greater_count

    # HEADLINE METRIC: a hit requires the true code to stand ALONE at rank 1.
    #
    # A heuristic returns a score over 4,699 candidates, not a name, so the true
    # code can tie with others at the top. This used to be credited
    # 1/(equal_count+1) -- the expected Top-1 under random tie-breaking, the
    # link-prediction convention. That made the baselines' number an expectation
    # while the LLM's was a plain 0/1 (it emits one name), so the two families
    # were not on one scale: 2,836 victims summed to 184.8 at x=5%, a figure
    # that counts no one. It also credited victims no attacker could resolve --
    # one tied with 2,954 other candidates still scored 0.03%.
    #
    # Under re-identification the question is binary (is this person named or
    # not), so a tie is a failure. Strict argmax is also the conservative
    # direction for a privacy-risk paper. The expectation is kept in
    # `top1_expected` for the supplement; nothing in the ranking changes --
    # Adamic-Adar reads 4.62/9.24/10.72% strict against 6.52/10.33/11.33%
    # expected, above the LLM either way.
    strict_top1 = 1.0 if (greater_count == 0 and equal_count == 0
                          and target_score > 0) else 0.0
    top1_expected = 1.0 if avg_rank <= 1.0 else (
        1.0 / (equal_count + 1) if greater_count == 0 and target_score > 0 else 0.0)
    top1 = strict_top1
    top3 = 1.0 if avg_rank <= 3.0 else (max(0.0, min(1.0, (3.0 - greater_count) / (equal_count + 1))) if greater_count < 3 and target_score > 0 else 0.0)
    top5 = 1.0 if avg_rank <= 5.0 else (max(0.0, min(1.0, (5.0 - greater_count) / (equal_count + 1))) if greater_count < 5 and target_score > 0 else 0.0)
    mrr = 1.0 / avg_rank

    return {
        "avg_rank": avg_rank,
        "min_rank": min_rank,
        "top1": top1,
        "top1_expected": top1_expected,
        "top3": top3,
        "top5": top5,
        "mrr": mrr,
        "target_score": target_score,
        "n_candidates_with_score": len(scored_cands),
    }


def run_baselines_on_dataset(rate_str: str, n_samples: int = None):
    """Run all baselines on the specified dataset rate."""
    print(f"\n{'='*75}")
    print(f"  RUNNING NON-LLM BASELINES ON FB15k-237 (Move Leak: {rate_str}%)")
    print(f"{'='*75}")

    triples, victims = load_dataset(rate_str)
    print(f"Loaded {len(triples):,} triples, {len(victims):,} victims.")

    print("Building graph indices and sparse transition matrix...")
    t0 = time.time()
    graph_struct = build_graph_structures(triples)
    print(f"Index & CSR built in {time.time() - t0:.2f}s. Total candidate IND nodes: {len(graph_struct['all_ind_nodes']):,}.")

    neighbors = graph_struct["neighbors"]
    incident_edges = graph_struct["incident_edges"]
    aa_weights = graph_struct["aa_weights"]
    ra_weights = graph_struct["ra_weights"]
    rel_idf = graph_struct["rel_idf"]
    all_ind_nodes = graph_struct["all_ind_nodes"]
    neighbor_to_ind = graph_struct["neighbor_to_ind"]
    rel_path_index = graph_struct["rel_path_index"]
    node_to_id = graph_struct["node_to_id"]
    P_T = graph_struct["P_T"]
    ind_indices = graph_struct["ind_indices"]
    n_nodes = P_T.shape[0]

    victim_items = list(victims.items())
    if n_samples:
        victim_items = victim_items[:n_samples]
        print(f"Subsampled to {len(victim_items)} victims for test run.")

    methods = [
        "random_global",
        "random_2hop",
        "common_neighbors",
        "jaccard",
        "adamic_adar",
        "resource_allocation",
        "relation_path",
        "weighted_anchor",
        "ppr_pagerank",
    ]
    metrics = {m: {"top1": [], "top1_expected": [], "top3": [], "top5": [],
               "mrr": [], "times": []} for m in methods}

    # Per-victim Top-1 outcome, kept alongside the aggregates so a later
    # analysis can bucket victims (e.g. by MID-key degree) and compare a
    # baseline against the LLM on the SAME victims. The aggregates above are
    # unchanged; this only records who each method got right.
    per_victim = {}

    # Seeded so the two random baselines draw the same guesses on every run;
    # every other method here is deterministic.
    rng = random.Random(4242)

    print(f"Evaluating {len(victim_items)} targets across 8 methods...")
    eval_t0 = time.time()

    for idx, (mid, true_ind) in enumerate(victim_items, 1):
        mid_neighbors = neighbors.get(mid, set())
        mid_edges = incident_edges.get(mid, [])

        # -------------------------------------------------------------
        # 1. Candidate space within 2 hops (sharing at least 1 anchor)
        # -------------------------------------------------------------
        two_hop_candidates = set()
        for neigh in mid_neighbors:
            two_hop_candidates.update(neighbor_to_ind.get(neigh, ()))
        n_2hop = len(two_hop_candidates)

        # Baselines 0a/0b: random guessing.
        #
        # These used to report the analytic expectation (1/n) for Top-1. Now
        # that every other method is scored on whether it actually named the
        # right code, these draw a real guess too, from a seeded RNG so the run
        # stays reproducible. Over 2,836 victims the sample mean lands within
        # noise of 1/n anyway; the point is that all nine numbers now mean the
        # same thing -- the share of victims a method actually named.
        n_all = len(all_ind_nodes)
        guess = all_ind_nodes[rng.randrange(n_all)]
        hit_global = 1.0 if guess == true_ind else 0.0
        metrics["random_global"]["top1"].append(hit_global)
        metrics["random_global"]["top3"].append(3.0 / n_all)
        metrics["random_global"]["top5"].append(5.0 / n_all)
        metrics["random_global"]["mrr"].append(math.log(n_all) / n_all)
        metrics["random_global"]["times"].append(0.0)

        # Baseline 0b: 2-Hop Candidate Random
        if n_2hop > 0:
            pool = sorted(two_hop_candidates)   # sorted => draw is reproducible
            hit_2hop = 1.0 if pool[rng.randrange(n_2hop)] == true_ind else 0.0
            metrics["random_2hop"]["top1"].append(hit_2hop)
            metrics["random_2hop"]["top3"].append(
                min(1.0, 3.0 / n_2hop) if true_ind in two_hop_candidates else 0.0)
            metrics["random_2hop"]["top5"].append(
                min(1.0, 5.0 / n_2hop) if true_ind in two_hop_candidates else 0.0)
            metrics["random_2hop"]["mrr"].append(math.log(max(2, n_2hop)) / n_2hop)
        else:
            metrics["random_2hop"]["top1"].append(0.0)
            metrics["random_2hop"]["top3"].append(0.0)
            metrics["random_2hop"]["top5"].append(0.0)
            metrics["random_2hop"]["mrr"].append(1.0 / n_all)
        metrics["random_2hop"]["times"].append(0.0)

        # -------------------------------------------------------------
        # 2. Common Neighbors (CN) & Jaccard
        # -------------------------------------------------------------
        t_cn_start = time.perf_counter()
        cn_scores = collections.defaultdict(float)
        for neigh in mid_neighbors:
            for cand in neighbor_to_ind.get(neigh, ()):
                cn_scores[cand] += 1.0

        jaccard_scores = {}
        deg_mid = len(mid_neighbors)
        for cand, cn in cn_scores.items():
            union = deg_mid + graph_struct["degree"][cand] - cn
            jaccard_scores[cand] = cn / union if union > 0 else 0.0

        t_cn = time.perf_counter() - t_cn_start

        res_cn = evaluate_ranking(cn_scores, true_ind, all_ind_nodes)
        metrics["common_neighbors"]["top1"].append(res_cn["top1"])
        metrics["common_neighbors"]["top1_expected"].append(res_cn["top1_expected"])
        metrics["common_neighbors"]["top3"].append(res_cn["top3"])
        metrics["common_neighbors"]["top5"].append(res_cn["top5"])
        metrics["common_neighbors"]["mrr"].append(res_cn["mrr"])
        metrics["common_neighbors"]["times"].append(t_cn)

        res_jac = evaluate_ranking(jaccard_scores, true_ind, all_ind_nodes)
        metrics["jaccard"]["top1"].append(res_jac["top1"])
        metrics["jaccard"]["top1_expected"].append(res_jac["top1_expected"])
        metrics["jaccard"]["top3"].append(res_jac["top3"])
        metrics["jaccard"]["top5"].append(res_jac["top5"])
        metrics["jaccard"]["mrr"].append(res_jac["mrr"])
        metrics["jaccard"]["times"].append(t_cn)

        # -------------------------------------------------------------
        # 3. Adamic-Adar Index (AA) & Resource Allocation (RA)
        # -------------------------------------------------------------
        t_aa_start = time.perf_counter()
        aa_scores = collections.defaultdict(float)
        ra_scores = collections.defaultdict(float)
        for neigh in mid_neighbors:
            w_aa = aa_weights.get(neigh, 0.0)
            w_ra = ra_weights.get(neigh, 0.0)
            for cand in neighbor_to_ind.get(neigh, ()):
                aa_scores[cand] += w_aa
                ra_scores[cand] += w_ra
        t_aa = time.perf_counter() - t_aa_start

        res_aa = evaluate_ranking(aa_scores, true_ind, all_ind_nodes)
        metrics["adamic_adar"]["top1"].append(res_aa["top1"])
        metrics["adamic_adar"]["top1_expected"].append(res_aa["top1_expected"])
        metrics["adamic_adar"]["top3"].append(res_aa["top3"])
        metrics["adamic_adar"]["top5"].append(res_aa["top5"])
        metrics["adamic_adar"]["mrr"].append(res_aa["mrr"])
        metrics["adamic_adar"]["times"].append(t_aa)

        res_ra = evaluate_ranking(ra_scores, true_ind, all_ind_nodes)
        metrics["resource_allocation"]["top1"].append(res_ra["top1"])
        metrics["resource_allocation"]["top1_expected"].append(res_ra["top1_expected"])
        metrics["resource_allocation"]["top3"].append(res_ra["top3"])
        metrics["resource_allocation"]["top5"].append(res_ra["top5"])
        metrics["resource_allocation"]["mrr"].append(res_ra["mrr"])
        metrics["resource_allocation"]["times"].append(t_aa)

        # -------------------------------------------------------------
        # 4. Relation-Path Matching & Weighted Anchor Matching
        # -------------------------------------------------------------
        t_rel_start = time.perf_counter()
        rel_scores = collections.defaultdict(float)
        weighted_anchor_scores = collections.defaultdict(float)

        for r, neigh, direction in mid_edges:
            w_r = rel_idf.get(r, 1.0)
            w_aa = aa_weights.get(neigh, 0.0)
            
            # Exact predicate-object match (fails if edge was MOVE without duplication)
            matched_cands = rel_path_index.get((r, neigh, direction), ())
            for cand in matched_cands:
                rel_scores[cand] += w_r

            # Weighted Anchor: weight shared anchor by relation selectivity * anchor rarity
            for cand in neighbor_to_ind.get(neigh, ()):
                weighted_anchor_scores[cand] += w_r * w_aa

        t_rel = time.perf_counter() - t_rel_start

        res_rel = evaluate_ranking(rel_scores, true_ind, all_ind_nodes)
        metrics["relation_path"]["top1"].append(res_rel["top1"])
        metrics["relation_path"]["top1_expected"].append(res_rel["top1_expected"])
        metrics["relation_path"]["top3"].append(res_rel["top3"])
        metrics["relation_path"]["top5"].append(res_rel["top5"])
        metrics["relation_path"]["mrr"].append(res_rel["mrr"])
        metrics["relation_path"]["times"].append(t_rel)

        res_wa = evaluate_ranking(weighted_anchor_scores, true_ind, all_ind_nodes)
        metrics["weighted_anchor"]["top1"].append(res_wa["top1"])
        metrics["weighted_anchor"]["top1_expected"].append(res_wa["top1_expected"])
        metrics["weighted_anchor"]["top3"].append(res_wa["top3"])
        metrics["weighted_anchor"]["top5"].append(res_wa["top5"])
        metrics["weighted_anchor"]["mrr"].append(res_wa["mrr"])
        metrics["weighted_anchor"]["times"].append(t_rel)

        # -------------------------------------------------------------
        # 5. Personalized PageRank (PPR via Sparse CSR)
        # -------------------------------------------------------------
        t_ppr_start = time.perf_counter()
        if mid in node_to_id:
            target_idx = node_to_id[mid]
            p_vec = compute_ppr_vector(P_T, target_idx, n_nodes, alpha=0.85, max_iter=15)
            # Extract scores for all IND candidates
            ind_scores_vec = p_vec[ind_indices]
            ppr_scores = {ind_name: float(ind_scores_vec[i]) for i, ind_name in enumerate(all_ind_nodes)}
        else:
            ppr_scores = {}
        t_ppr = time.perf_counter() - t_ppr_start

        res_ppr = evaluate_ranking(ppr_scores, true_ind, all_ind_nodes)
        metrics["ppr_pagerank"]["top1"].append(res_ppr["top1"])
        metrics["ppr_pagerank"]["top1_expected"].append(res_ppr["top1_expected"])
        metrics["ppr_pagerank"]["top3"].append(res_ppr["top3"])
        metrics["ppr_pagerank"]["top5"].append(res_ppr["top5"])
        metrics["ppr_pagerank"]["mrr"].append(res_ppr["mrr"])
        metrics["ppr_pagerank"]["times"].append(t_ppr)

        # Record this victim's Top-1 outcome per method. The two random
        # baselines score an expectation rather than a hit, so a threshold
        # would be meaningless for them; they are stored as the expectation
        # itself and should be read as such, not counted as hits.
        per_victim[mid] = {m: metrics[m]["top1"][-1] for m in methods}

        if idx % 500 == 0 or idx == len(victim_items):
            elapsed = time.time() - eval_t0
            print(f"  Processed {idx}/{len(victim_items)} victims ({elapsed:.1f}s, {elapsed/idx*1000:.1f} ms/victim)...")

    # Aggregate summaries
    summary = {}
    for m in methods:
        n_eval = len(metrics[m]["top1"])
        if n_eval == 0:
            continue
        summary[m] = {
            "n_evaluated": n_eval,
            # Headline: share of victims the method actually named (strict argmax).
            "top1_acc": float(np.mean(metrics[m]["top1"])),
            "n_hits": int(sum(metrics[m]["top1"])),
            # Supplement only: the old tie-expectation figure, kept so the
            # stricter headline can be reported against it.
            "top1_acc_expected": (float(np.mean(metrics[m]["top1_expected"]))
                                  if metrics[m]["top1_expected"] else None),
            "top3_recall": float(np.mean(metrics[m]["top3"])),
            "top5_recall": float(np.mean(metrics[m]["top5"])),
            "mrr": float(np.mean(metrics[m]["mrr"])),
            "avg_latency_ms": float(np.mean(metrics[m]["times"]) * 1000.0),
        }

    # Print nicely formatted table
    print(f"\n{'-'*85}")
    print(f"  SUMMARY RESULTS (FB15k-237 @ {rate_str}% MOVE ERROR, N={len(victim_items)})")
    print(f"{'-'*85}")
    print(f"  {'Method':<25} {'Top-1 Acc':>10} {'Top-3 Rec':>10} {'Top-5 Rec':>10} {'MRR':>10} {'Latency':>10}")
    print(f"  {'-'*83}")
    for m in methods:
        if m not in summary:
            continue
        s = summary[m]
        print(f"  {m:<25} {s['top1_acc']:>9.2%} {s['top3_recall']:>9.2%} {s['top5_recall']:>9.2%} {s['mrr']:>10.4f} {s['avg_latency_ms']:>8.2f}ms")
    print(f"{'-'*85}\n")

    return summary, per_victim


def main():
    parser = argparse.ArgumentParser(description="Run Non-LLM Baselines on Knowledge Graph")
    parser.add_argument("--rate", type=str, default="05", choices=["00", "05", "10", "15"],
                        help="Leak rate (05, 10, 15)")
    parser.add_argument("--all_rates", action="store_true",
                        help="Run over all three leak rates (05, 10, 15)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Sample limit for quick tests")
    parser.add_argument("--test", action="store_true",
                        help="Quick test run on 50 samples")
    parser.add_argument("--out_json", type=str, default=None,
                        help="Custom path to save summary JSON")
    args = parser.parse_args()

    if args.test:
        args.n_samples = 50

    results_dir = os.path.join(BASE_DIR, "deanon_results")
    os.makedirs(results_dir, exist_ok=True)

    rates = ["05", "10", "15"] if args.all_rates else [args.rate]
    all_summaries = {}

    all_per_victim = {}
    for r in rates:
        summary, per_victim = run_baselines_on_dataset(r, n_samples=args.n_samples)
        all_summaries[f"rate_{r}"] = summary
        all_per_victim[f"rate_{r}"] = per_victim

    out_file = args.out_json or os.path.join(results_dir, "non_llm_baselines_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Saved complete baseline results to {out_file}")

    # Per-victim outcomes, written beside the summary. case_bands.py needs
    # these to compare a baseline against the LLM on the same victims; the
    # summary alone only gives corpus-wide averages.
    pv_file = os.path.splitext(out_file)[0].replace("_summary", "") + "_per_victim.json"
    with open(pv_file, "w", encoding="utf-8") as f:
        json.dump(all_per_victim, f, indent=2)
    print(f"Saved per-victim baseline outcomes to {pv_file}")


if __name__ == "__main__":
    main()
