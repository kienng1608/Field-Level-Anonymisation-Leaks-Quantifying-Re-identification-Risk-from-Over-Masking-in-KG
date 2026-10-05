"""Two facts the main text needs at x=0 (the masked corpus, nothing misfiled):

  1. what share of subjects have an empty candidate pool  (V-A says "most")
  2. why the handful of hits happen -- which relation on the identified key
     reaches the anchor the masked key also touches.

Scored with the project's own structures, so the hit counts must reproduce the
published x=0 column: Jac 0.25, CN 0.11, AA/RA/WA/PPR 0.18.
"""
import collections
import io
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
sys.path.insert(0, os.path.join(BASE, "baselines"))
sys.path.insert(0, os.path.join(BASE, "corpus"))
import run_non_llm_baselines as NB

tri = []
for line in io.open("data/FB15k-237-id-masked/train.txt", encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    if len(p) == 3:
        tri.append(tuple(p))
vic = json.load(io.open("data/FB15k-237-id-masked/victims.json", encoding="utf-8"))

G = NB.build_graph_structures(tri)
nb, degree, inc = G["neighbors"], G["degree"], G["incident_edges"]
aaw, raw, idf = G["aa_weights"], G["ra_weights"], G["rel_idf"]
n2i, inds = G["neighbor_to_ind"], G["all_ind_nodes"]

empty = 0
hits = collections.defaultdict(list)
for mid, ti in vic.items():
    mn = nb.get(mid, set())
    cn, aa, ra = (collections.defaultdict(float) for _ in range(3))
    for a in mn:
        for c in n2i.get(a, ()):
            if c == mid:
                continue
            cn[c] += 1.0
            aa[c] += aaw.get(a, 0.0)
            ra[c] += raw.get(a, 0.0)
    if not cn:
        empty += 1
        continue
    jac = {c: v / (len(mn) + degree[c] - v) for c, v in cn.items()
           if len(mn) + degree[c] - v > 0}
    for name, sc in (("CN", cn), ("Jac", jac), ("AA", aa), ("RA", ra)):
        if NB.evaluate_ranking(sc, ti, inds)["top1"] == 1.0:
            hits[name].append((mid, ti))

N = len(vic)
print("empty candidate pool: %d / %d = %.1f%%" % (empty, N, 100.0 * empty / N))
for k in ("CN", "Jac", "AA", "RA"):
    print("  %-4s %d hits = %.2f%%" % (k, len(hits[k]), 100.0 * len(hits[k]) / N))

print("\nmechanism, over the union of those hits:")
seen = {}
for k in hits:
    for mid, ti in hits[k]:
        seen[mid] = ti
rel_ident = collections.Counter()
rel_masked = collections.Counter()
for mid, ti in seen.items():
    shared = set(nb.get(mid, ())) & set(nb.get(ti, ()))
    for r, a, _d in inc.get(mid, []):
        if a in shared:
            rel_masked[r] += 1
    for r, a, _d in inc.get(ti, []):
        if a in shared:
            rel_ident[r] += 1
print("  %d subjects" % len(seen))
print("  relation on the MASKED key reaching the shared anchor:")
for r, n in rel_masked.most_common(6):
    print("     %-70s %d" % (r[:70], n))
print("  relation on the IDENTIFIED key reaching the same anchor:")
for r, n in rel_ident.most_common(6):
    print("     %-70s %d" % (r[:70], n))
