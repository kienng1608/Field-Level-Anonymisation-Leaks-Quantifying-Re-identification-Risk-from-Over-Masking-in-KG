"""Does connectivity predict exposure when the leak is held fixed?

Corpora: fixed-k (k = 1, 2, 3 edges moved per subject) and, as reference, the
proportional x = 15% corpus. For every subject:
  deg   ordinary degree of the identified key BEFORE any leak (masked corpus)
  L     realised leak: edges on m_v in the leaked corpus minus in the masked one
  hits  strict top-1 for CN, Jaccard, AA, RA over all person codes
Scoring mirrors baselines/run_non_llm_baselines.py (verified below against the
published per-subject results on the x = 15% corpus).
Reports hit rates by degree quartile (quartiles fixed on pre-leak degree) and a
logistic fit hit ~ log2(deg), on all subjects and on those with L exactly k.
"""
import collections
import io
import json
import math
import os

import numpy as np
import statsmodels.api as sm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RESULTS = os.path.join(BASE, "deanon_results", "analysis")
os.makedirs(RESULTS, exist_ok=True)
HERE = RESULTS
METHODS = ["CN", "Jac", "AA", "RA"]


def load(name):
    tr = []
    for line in io.open(os.path.join(DATA, name, "train.txt"), encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) == 3:
            tr.append(tuple(p))
    return tr


def top1(sc, ti):
    ts = sc.get(ti, 0.0)
    return int(ts > 0 and all(v < ts for k, v in sc.items() if k != ti))


vic = json.load(io.open(os.path.join(DATA, "FB15k-237-id-masked", "victims.json"),
                        encoding="utf-8"))
base = load("FB15k-237-id-masked")
deg0 = collections.Counter()
onm0 = collections.Counter()
for h, r, t in base:
    for e in (h, t):
        deg0[e] += 1
        onm0[e] += 1
subj_deg = {m: deg0[c] for m, c in vic.items()}


def score(name):
    tr = load(name)
    nb = collections.defaultdict(set)
    onm = collections.Counter()
    for h, r, t in tr:
        nb[h].add(t)
        nb[t].add(h)
        onm[h] += 1
        onm[t] += 1
    aaw = {k: 1.0 / math.log(max(2, len(v))) for k, v in nb.items()}
    raw = {k: 1.0 / max(1, len(v)) for k, v in nb.items()}
    n2i = collections.defaultdict(list)
    for node in nb:
        if node.startswith("IND_"):
            for a in nb[node]:
                n2i[a].append(node)
    rows = {}
    for mid, ti in vic.items():
        am = nb.get(mid, set())
        cn = collections.defaultdict(float)
        aa = collections.defaultdict(float)
        ra = collections.defaultdict(float)
        for a in am:
            for c in n2i.get(a, ()):
                cn[c] += 1.0
                aa[c] += aaw[a]
                ra[c] += raw[a]
        jac = {c: v / len(am | nb[c]) for c, v in cn.items()}
        rows[mid] = {"deg": subj_deg[mid], "L": onm[mid] - onm0[mid],
                     "CN": top1(cn, ti), "Jac": top1(jac, ti),
                     "AA": top1(aa, ti), "RA": top1(ra, ti)}
    return rows


# ---- scorer check against the published per-subject results ----
pub = json.load(io.open(os.path.join(BASE, "deanon_results",
                                     "non_llm_baselines_per_victim.json"),
                        encoding="utf-8"))["rate_15"]
ref = score("FB15k-237-id-move15")
key = {"CN": "common_neighbors", "Jac": "jaccard", "AA": "adamic_adar",
       "RA": "resource_allocation"}
print("scorer check on x=15%: " + ", ".join(
    "%s %d/%d" % (m, sum(ref[s][m] == int(pub[s][key[m]]) for s in ref), len(ref))
    for m in METHODS))

degs = np.array([subj_deg[m] for m in sorted(vic)])
qcut = np.quantile(degs, [0.25, 0.5, 0.75])
print("pre-leak degree quartile cut points:", [int(q) for q in qcut])


def quart(d):
    return int(np.searchsorted(qcut, d, side="right"))


def report(tag, rows):
    subs = sorted(rows)
    Ls = collections.Counter(rows[s]["L"] for s in subs)
    print("\n=== %s: %d subjects, realised leak L: %s"
          % (tag, len(subs), dict(sorted(Ls.items()))))
    print("  hit rate by degree quartile (Q1 least connected):")
    for m in METHODS:
        by = collections.defaultdict(list)
        for s in subs:
            by[quart(rows[s]["deg"])].append(rows[s][m])
        print("    %-4s %s   overall %5.2f%%" % (
            m, "  ".join("Q%d %5.2f%%" % (q + 1, 100 * np.mean(by[q]))
                         for q in range(4)),
            100 * np.mean([rows[s][m] for s in subs])))
    return subs


def logit(tag, rows, subs, extra_L=False):
    x = np.log2([rows[s]["deg"] for s in subs])
    X = np.column_stack([x, [rows[s]["L"] for s in subs]]) if extra_L else x[:, None]
    X = sm.add_constant(X)
    out = []
    for m in METHODS:
        y = np.array([rows[s][m] for s in subs])
        if y.sum() < 5:
            out.append("%s n/a (%d hits)" % (m, y.sum()))
            continue
        f = sm.Logit(y, X).fit(disp=0)
        b, se = f.params[1], f.bse[1]
        out.append("%s OR %.2f [%.2f, %.2f] p=%.2g" % (
            m, math.exp(b), math.exp(b - 1.96 * se), math.exp(b + 1.96 * se),
            f.pvalues[1]))
    print("  logit hit ~ log2(degree)%s  [%s]:" % (" + L" if extra_L else "", tag))
    for o in out:
        print("    " + o)


results = {"x15": ref}
subs = report("proportional x=15%", ref)
logit("all", ref, subs)
logit("all", ref, subs, extra_L=True)

for k in (1, 2, 3):
    rows = score("FB15k-237-id-fixk%d" % k)
    results["k%d" % k] = rows
    subs = report("fixed k=%d" % k, rows)
    logit("all subjects", rows, subs, extra_L=True)
    exact = [s for s in subs if rows[s]["L"] == k]
    print("  subjects with L exactly %d: %d" % (k, len(exact)))
    logit("L == k only", rows, exact)

json.dump(results, io.open(os.path.join(HERE, "m1_fixedk.json"), "w",
                           encoding="utf-8"))
