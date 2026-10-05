"""What does the neighbour-exclusion rule do to every index?

Rule: drop from the ranking every person code adjacent to m_v. It never removes
the answer (no subject's c_v is adjacent to its m_v, checked in all corpora).
Scores are rebuilt with the project's own functions; the 'off' column must
reproduce the published hits.
"""
import collections
import io
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "baselines"))
sys.path.insert(0, os.path.join(BASE, "corpus"))
import run_non_llm_baselines as NB

PUB = {"CN": "common_neighbors", "Jac": "jaccard", "AA": "adamic_adar",
       "RA": "resource_allocation", "WA": "weighted_anchor", "PPR": "ppr_pagerank"}
METHODS = ["PPR", "Jac", "CN", "WA", "AA", "RA"]
pub = json.load(io.open(os.path.join(BASE, "deanon_results",
                                     "non_llm_baselines_per_victim.json"),
                        encoding="utf-8"))

for rate, corpus in (("00", "FB15k-237-id-masked"), ("05", "FB15k-237-id-move05"),
                     ("10", "FB15k-237-id-move10"), ("15", "FB15k-237-id-move15")):
    d = os.path.join(BASE, "data", corpus)
    vic = json.load(io.open(os.path.join(d, "victims.json"), encoding="utf-8"))
    tr = []
    for line in io.open(os.path.join(d, "train.txt"), encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) == 3:
            tr.append(tuple(p))
    G = NB.build_graph_structures(tr)
    nb, inc, degree = G["neighbors"], G["incident_edges"], G["degree"]
    idf, aaw, raw = G["rel_idf"], G["aa_weights"], G["ra_weights"]
    n2i, inds = G["neighbor_to_ind"], G["all_ind_nodes"]
    P_T, node_to_id, ind_idx = G["P_T"], G["node_to_id"], G["ind_indices"]
    n_nodes = P_T.shape[0]

    hits = {"off": collections.Counter(), "on": collections.Counter()}
    check = collections.Counter()
    n_adj = 0
    for mid, ti in vic.items():
        mn = nb.get(mid, set())
        adj = {c for c in mn if c.startswith("IND_")}
        n_adj += bool(adj)
        cn = collections.defaultdict(float)
        aa = collections.defaultdict(float)
        ra = collections.defaultdict(float)
        for a in mn:
            for c in n2i.get(a, ()):
                cn[c] += 1.0
                aa[c] += aaw.get(a, 0.0)
                ra[c] += raw.get(a, 0.0)
        jac = {c: v / (len(mn) + degree[c] - v) for c, v in cn.items()
               if len(mn) + degree[c] - v > 0}
        wa = collections.defaultdict(float)
        for r, a, _d in inc.get(mid, []):
            w = idf.get(r, 1.0) * aaw.get(a, 0.0)
            for c in n2i.get(a, ()):
                wa[c] += w
        if mid in node_to_id:
            vec = NB.compute_ppr_vector(P_T, node_to_id[mid], n_nodes, 0.85, 15)
            sc = vec[ind_idx]
            ppr = {inds[i]: float(sc[i]) for i in range(len(inds))}
        else:
            ppr = {}
        for name, s in (("CN", cn), ("Jac", jac), ("AA", aa), ("RA", ra),
                        ("WA", wa), ("PPR", ppr)):
            h_off = int(NB.evaluate_ranking(s, ti, inds)["top1"] == 1.0)
            s_on = {c: v for c, v in s.items() if c not in adj}
            h_on = int(NB.evaluate_ranking(s_on, ti, inds)["top1"] == 1.0)
            hits["off"][name] += h_off
            hits["on"][name] += h_on
            check[name] += h_off == int(pub["rate_" + rate][mid][PUB[name]]) \
                if "rate_" + rate in pub else 0
    N = len(vic)
    print("\nx=%s%%: subjects with a person code adjacent to m_v: %d (%.1f%%)"
          % (rate, n_adj, 100.0 * n_adj / N))
    if "rate_" + rate in pub:
        print("  scorer check: " + ", ".join("%s %d/%d" % (m, check[m], N)
                                            for m in METHODS))
    for m in METHODS:
        a, b = hits["off"][m], hits["on"][m]
        print("  %-4s off %5.2f%%  on %5.2f%%  change %+5.2f"
              % (m, 100.0 * a / N, 100.0 * b / N, 100.0 * (b - a) / N))
