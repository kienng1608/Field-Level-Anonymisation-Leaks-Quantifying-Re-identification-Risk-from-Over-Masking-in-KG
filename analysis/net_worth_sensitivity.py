"""Sensitivity to the net-worth relation (outside GDPR Art. 9).

1. Regression: rebuild the masked corpus and the x=15% corpus with the real
   pipeline (corpus/step2_conditional_mask.py, corpus/step3_move_leak.py, seed
   4242) and check they are byte-identical to data/FB15k-237-id-masked and
   data/FB15k-237-id-move15.
2. Variant: same pipeline with net_worth removed from SENSITIVE_RELATIONS in
   both steps (it becomes an ordinary relation, movable like any other).
3. Score CN, Jaccard, AA, RA, WA, PPR (strict top-1) on x=0 and x=15% for both,
   with the scorer that reproduces the published per-subject results.
"""
import collections
import filecmp
import importlib
import io
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "deanon_results", "analysis")
os.makedirs(RESULTS, exist_ok=True)
HERE = RESULTS
OUT = os.path.join(HERE, "nw_corpora")
os.chdir(BASE)
sys.path.insert(0, os.path.join(BASE, "baselines"))
sys.path.insert(0, os.path.join(BASE, "corpus"))
import run_non_llm_baselines as NB  # noqa: E402

NW = "/base/schemastaging/person_extra/net_worth./measurement_unit/dated_money_value/currency"


def build(tag, drop):
    s2 = importlib.reload(importlib.import_module("step2_conditional_mask"))
    s3 = importlib.reload(importlib.import_module("step3_move_leak"))
    if drop:
        s2.SENSITIVE_RELATIONS = [r for r in s2.SENSITIVE_RELATIONS if r != NW]
        s2.SENSITIVE_VICTIM_POSITION = {k: v for k, v in s2.SENSITIVE_VICTIM_POSITION.items() if k != NW}
        s3.SENSITIVE_RELATIONS = [r for r in s3.SENSITIVE_RELATIONS if r != NW]
    masked = os.path.join(OUT, tag + "-masked")
    move15 = os.path.join(OUT, tag + "-move15")
    sys.argv = ["step2", "--input_dir", "data/FB15k-237-id-base", "--output_dir", masked]
    s2.main()
    sys.argv = ["step3", "--input_dir", masked, "--output_dir", move15,
                "--rate", "0.15", "--seed", "4242"]
    s3.main()
    return masked, move15


def load(d):
    tri = []
    for line in io.open(os.path.join(d, "train.txt"), encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) == 3:
            tri.append(tuple(p))
    return tri, json.load(io.open(os.path.join(d, "victims.json"), encoding="utf-8"))


def score(d):
    tr, vic = load(d)
    G = NB.build_graph_structures(tr)
    nb, inc, degree = G["neighbors"], G["incident_edges"], G["degree"]
    idf, aaw, raw = G["rel_idf"], G["aa_weights"], G["ra_weights"]
    n2i, inds = G["neighbor_to_ind"], G["all_ind_nodes"]
    P_T, node_to_id, ind_idx = G["P_T"], G["node_to_id"], G["ind_indices"]
    hits = collections.Counter()
    per = {}
    for mid, ti in vic.items():
        mn = nb.get(mid, set())
        cn, aa, ra = (collections.defaultdict(float) for _ in range(3))
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
            vec = NB.compute_ppr_vector(P_T, node_to_id[mid], P_T.shape[0], 0.85, 15)
            sc = vec[ind_idx]
            ppr = {inds[i]: float(sc[i]) for i in range(len(inds))}
        else:
            ppr = {}
        row = {}
        for name, s in (("CN", cn), ("Jac", jac), ("AA", aa), ("RA", ra), ("WA", wa), ("PPR", ppr)):
            row[name] = int(NB.evaluate_ranking(s, ti, inds)["top1"] == 1.0)
            hits[name] += row[name]
        per[mid] = row
    return len(vic), hits, per


os.makedirs(OUT, exist_ok=True)
print("=== 1. regression: rebuild the published corpora ===")
m_chk, x_chk = build("check", drop=False)
for mine, pub in ((m_chk, "data/FB15k-237-id-masked"), (x_chk, "data/FB15k-237-id-move15")):
    for f in ("train.txt", "victims.json"):
        same = filecmp.cmp(os.path.join(mine, f), os.path.join(pub, f), shallow=False)
        print("  %-28s %-13s identical: %s" % (pub, f, same))

print("\n=== 2. variant without net worth ===")
m_nw, x_nw = build("nonw", drop=True)

print("\n=== 3. scores (strict top-1) ===")
res = {}
for label, d in (("published x=0", "data/FB15k-237-id-masked"),
                 ("published x=15%", "data/FB15k-237-id-move15"),
                 ("no net worth x=0", m_nw),
                 ("no net worth x=15%", x_nw)):
    n, h, per = score(d)
    res[label] = {"subjects": n, **{k: 100.0 * v / n for k, v in h.items()}}
    print("  %-20s subjects %5d  " % (label, n) +
          "  ".join("%s %5.2f%%" % (k, 100.0 * h[k] / n) for k in ("CN", "Jac", "AA", "RA", "WA", "PPR")))
    res[label]["per"] = per

pub15, nw15 = res["published x=15%"], res["no net worth x=15%"]
common = set(pub15["per"]) & set(nw15["per"])
print("\n  subjects dropped: %d; kept: %d" % (pub15["subjects"] - nw15["subjects"], len(common)))
for k in ("AA", "RA", "WA"):
    a = sum(pub15["per"][m][k] for m in common)
    b = sum(nw15["per"][m][k] for m in common)
    print("  %s on the %d subjects common to both: published %.2f%%, without net worth %.2f%%"
          % (k, len(common), 100.0 * a / len(common), 100.0 * b / len(common)))
json.dump({k: {kk: vv for kk, vv in v.items() if kk != "per"} for k, v in res.items()},
          io.open(os.path.join(HERE, "nw_sensitivity.json"), "w", encoding="utf-8"), indent=1)
