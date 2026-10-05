"""Separate the two ways Weighted Anchor differs from Adamic-Adar:
  AA        each shared anchor once,            weight 1/log deg(a)
  AA-mult   each triple of m_v to a shared anchor, weight 1/log deg(a)   (new)
  WA        each triple of m_v to a shared anchor, weight idf(r)/log deg(a)
AA -> AA-mult isolates per-triple counting; AA-mult -> WA isolates idf.
Scoring mirrors baselines/run_non_llm_baselines.py: candidates are all person
codes, strict top-1, ties and zero scores fail. AA and WA are recomputed as a
check against the published per-subject results.
"""
import collections
import io
import json
import math
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pub = json.load(io.open(os.path.join(
    BASE, "deanon_results", "non_llm_baselines_per_victim.json"),
    encoding="utf-8"))


def top1(sc, ti):
    ts = sc.get(ti, 0.0)
    return int(ts > 0 and all(v < ts for k, v in sc.items() if k != ti))


for rate in ("05", "10", "15"):
    vic = json.load(io.open(os.path.join(
        BASE, "data", "FB15k-237-id-move%s" % rate, "victims.json"),
        encoding="utf-8"))
    tr = []
    for line in io.open(os.path.join(BASE, "data", "FB15k-237-id-move%s" % rate,
                                     "train.txt"), encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) == 3:
            tr.append(tuple(p))
    nb = collections.defaultdict(set)
    inc = collections.defaultdict(list)
    rf = collections.Counter()
    for h, r, t in tr:
        nb[h].add(t)
        nb[t].add(h)
        inc[h].append((r, t))
        inc[t].append((r, h))
        rf[r] += 1
    n = len(tr)
    idf = {r: math.log((n + 1.0) / (c + 1.0)) for r, c in rf.items()}
    aaw = {k: 1.0 / math.log(max(2, len(v))) for k, v in nb.items()}
    n2i = collections.defaultdict(list)
    for node in nb:
        if node.startswith("IND_"):
            for a in nb[node]:
                n2i[a].append(node)

    hits = collections.Counter()
    agree = collections.Counter()
    for mid, ti in vic.items():
        aa = collections.defaultdict(float)
        for a in nb.get(mid, ()):
            for c in n2i.get(a, ()):
                aa[c] += aaw[a]
        mult = collections.defaultdict(float)
        wa = collections.defaultdict(float)
        for r, a in inc.get(mid, ()):
            for c in n2i.get(a, ()):
                mult[c] += aaw[a]
                wa[c] += idf[r] * aaw[a]
        h_aa, h_m, h_wa = top1(aa, ti), top1(mult, ti), top1(wa, ti)
        hits["AA"] += h_aa
        hits["AA-mult"] += h_m
        hits["WA"] += h_wa
        agree["AA"] += h_aa == int(pub["rate_" + rate][mid]["adamic_adar"])
        agree["WA"] += h_wa == int(pub["rate_" + rate][mid]["weighted_anchor"])
    N = len(vic)
    print("x=%s%%  AA %.2f%%  AA-mult %.2f%%  WA %.2f%%   (agreement with published: AA %d/%d, WA %d/%d)"
          % (rate, 100.0 * hits["AA"] / N, 100.0 * hits["AA-mult"] / N,
             100.0 * hits["WA"] / N, agree["AA"], N, agree["WA"], N))
