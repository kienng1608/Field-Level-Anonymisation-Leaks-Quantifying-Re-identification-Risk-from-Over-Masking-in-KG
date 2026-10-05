"""Precision at a given coverage -- does an attacker know when it is right?

Heuristics: confidence = margin between the top two scores over all 4,699
person codes (a tie gives margin 0 and counts as declining). Ranking subjects by
margin gives a precision-coverage curve. Scores are rebuilt with the project's
own build_graph_structures / compute_ppr_vector / evaluate_ranking, and the hits
are checked against the published per-subject results.

Agent: it declines by returning no prediction, so its coverage is the share of
subjects it answers; precision = strict hits / answers. Where the last round
carries a stated confidence, it also gives the agent its own curve.
"""
import collections
import glob
import io
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "baselines"))
sys.path.insert(0, os.path.join(BASE, "corpus"))
import run_non_llm_baselines as NB

RESULTS = os.path.join(BASE, "deanon_results", "analysis")
os.makedirs(RESULTS, exist_ok=True)
HERE = RESULTS
PAT = {"05": "exp_all_gemini25_batch*.json", "10": "exp10_gemini25_batch*.json",
       "15": "exp15_gemini25_batch*.json"}
PUB = {"CN": "common_neighbors", "Jac": "jaccard", "AA": "adamic_adar",
       "RA": "resource_allocation", "WA": "weighted_anchor", "PPR": "ppr_pagerank"}
METHODS = ["PPR", "Jac", "CN", "WA", "AA", "RA"]

pub = json.load(io.open(os.path.join(BASE, "deanon_results",
                                     "non_llm_baselines_per_victim.json"),
                        encoding="utf-8"))
summary = {}

for rate in ("05", "10", "15"):
    d = os.path.join(BASE, "data", "FB15k-237-id-move%s" % rate)
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

    rows = {}
    check = collections.Counter()
    for mid, ti in vic.items():
        mn = nb.get(mid, set())
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
        for r, a, _dir in inc.get(mid, []):
            w = idf.get(r, 1.0) * aaw.get(a, 0.0)
            for c in n2i.get(a, ()):
                wa[c] += w
        if mid in node_to_id:
            vec = NB.compute_ppr_vector(P_T, node_to_id[mid], n_nodes, 0.85, 15)
            sc = vec[ind_idx]
            ppr = {inds[i]: float(sc[i]) for i in range(len(inds))}
        else:
            ppr = {}
        out = {}
        for name, s in (("CN", cn), ("Jac", jac), ("AA", aa), ("RA", ra),
                        ("WA", wa), ("PPR", ppr)):
            hit = int(NB.evaluate_ranking(s, ti, inds)["top1"] == 1.0)
            top = sorted(s.values(), reverse=True)[:2] + [0.0, 0.0]
            out[name] = (hit, top[0] - top[1])
            check[name] += hit == int(pub["rate_" + rate][mid][PUB[name]])
        rows[mid] = out

    ag, conf = {}, {}
    for f in sorted(glob.glob(os.path.join(BASE, "deanon_results", PAT[rate]))):
        if "partial" in f:
            continue
        for e in json.load(io.open(f, encoding="utf-8"))["results"]:
            answered = bool(e.get("predictions"))
            ag[e["mid"]] = (answered,
                            int(bool(e.get("match")) and e.get("rank") == 1))
            c = (e.get("rounds") or [{}])[-1].get("top1_confidence")
            if answered and c is not None:
                conf[e["mid"]] = float(c)

    N = len(vic)
    n_ans = sum(1 for a, h in ag.values() if a)
    a_hits = sum(h for a, h in ag.values() if a)
    print("\n=== x=%s%%  (scorer check vs published: %s)"
          % (rate, ", ".join("%s %d/%d" % (m, check[m], N) for m in METHODS)))
    print("  agent answers %d of %d (coverage %.1f%%), %d correct: precision %.1f%%;"
          " stated confidence present for %d answers"
          % (n_ans, N, 100.0 * n_ans / N, a_hits, 100.0 * a_hits / n_ans,
             len(conf)))
    marks = {"1%": round(0.01 * N), "5%": round(0.05 * N),
             "10%": round(0.10 * N), "agent's": n_ans}
    print("  %-4s %8s | %s" % ("idx", "max cov", "  ".join(
        "prec@%-7s" % k for k in marks)))
    res = {}
    for m in METHODS:
        order = sorted(vic, key=lambda s: -rows[s][m][1])
        pos = [s for s in order if rows[s][m][1] > 0]
        cells = []
        for k, n in marks.items():
            if n <= len(pos):
                p = 100.0 * sum(rows[s][m][0] for s in pos[:n]) / n
                cells.append("%6.1f%%   " % p)
                res[k] = p
            else:
                cells.append("   --     ")
        print("  %-4s %7.1f%% | %s" % (m, 100.0 * len(pos) / N, "  ".join(cells)))
        summary.setdefault(rate, {})[m] = res.copy()
    summary[rate]["agent"] = {"coverage": 100.0 * n_ans / N,
                              "precision": 100.0 * a_hits / n_ans}
    # the agent's own curve, by stated confidence, among answers that carry one
    if len(conf) >= 0.9 * n_ans:
        order = sorted(conf, key=lambda s: -conf[s])
        for k in ("1%", "5%", "10%"):
            n = marks[k]
            if n <= len(order):
                p = 100.0 * sum(ag[s][1] for s in order[:n]) / n
                print("  agent by stated confidence: precision@%s %.1f%%" % (k, p))

json.dump(summary, io.open(os.path.join(HERE, "m4_precision.json"), "w",
                           encoding="utf-8"), indent=1)
