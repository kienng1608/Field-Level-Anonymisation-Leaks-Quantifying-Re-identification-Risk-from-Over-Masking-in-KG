"""Score every baseline on the evidence the agent itself retrieved, so all ten
attackers see the same subgraph for each subject.

Extends baselines/s3_budget_matched.py, which scored only AA and RA, to the full
set: CN, Jaccard, AA, RA, Weighted Anchor, PPR, plus the three controls.
Scoring rules are copied from baselines/run_non_llm_baselines.py so the numbers are
comparable with the full-graph run:
  - anchors are the nodes a key touches, person codes included
  - deg(a) = number of distinct adjacent nodes, computed inside the subgraph
  - idf(r) = log((|T|+1)/(count+1)), computed inside the subgraph
  - strict top-1; a tie counts as a failure
"""
import argparse
import collections
import glob
import io
import json
import math
import os
import random
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "baselines"))
sys.path.insert(0, os.path.join(BASE, "corpus"))
import s3_budget_matched as S


def score_all(triples, target, true_ind, rng):
    """All nine baselines over this subgraph. Returns {name: 0/1}."""
    nb = collections.defaultdict(set)
    inc = collections.defaultdict(list)
    relf = collections.Counter()
    for h, r, t in triples:
        nb[h].add(t)
        nb[t].add(h)
        inc[h].append((r, t, "out"))
        inc[t].append((r, h, "in"))
        relf[r] += 1
    deg = {k: len(v) for k, v in nb.items()}
    n_tr = len(triples)
    idf = {r: math.log((n_tr + 1.0) / (c + 1.0)) for r, c in relf.items()}

    cand = [n for n in nb if n.startswith("IND_") and n != target]
    if not cand:
        return {k: 0 for k in ("CN", "Jac", "AA", "RA", "WA", "PPR",
                               "rand_g", "rand_2h", "relpath")}, 0

    n2i = collections.defaultdict(list)
    for c in cand:
        for a in nb[c]:
            n2i[a].append(c)

    anchors = nb.get(target, set())
    two_hop = sorted({c for a in anchors for c in n2i.get(a, ())})

    aaw = {n: 1.0 / math.log(max(2, d)) for n, d in deg.items()}
    raw = {n: 1.0 / max(1, d) for n, d in deg.items()}

    cn = collections.defaultdict(float)
    jac = collections.defaultdict(float)
    aa = collections.defaultdict(float)
    ra = collections.defaultdict(float)
    for a in anchors:
        for c in n2i.get(a, ()):
            cn[c] += 1.0
            aa[c] += aaw.get(a, 0.0)
            ra[c] += raw.get(a, 0.0)
    for c in cn:
        u = len(anchors | nb.get(c, set()))
        jac[c] = cn[c] / u if u else 0.0

    wa = collections.defaultdict(float)
    rp = collections.defaultdict(float)
    rp_index = collections.defaultdict(list)
    for c in cand:
        for r, nn, d in inc[c]:
            rp_index[(r, nn, d)].append(c)
    for r, nn, d in inc.get(target, ()):
        w = idf.get(r, 1.0)
        for c in n2i.get(nn, ()):
            wa[c] += w * aaw.get(nn, 0.0)
        for c in rp_index.get((r, nn, d), ()):
            rp[c] += w

    # Personalized PageRank restricted to this subgraph
    nodes = sorted(nb)
    idx = {n: i for i, n in enumerate(nodes)}
    pr = [0.0] * len(nodes)
    if target in idx:
        pr[idx[target]] = 1.0
        for _ in range(30):
            nxt = [0.0] * len(nodes)
            for n, i in idx.items():
                if pr[i] == 0.0:
                    continue
                share = 0.85 * pr[i] / max(1, len(nb[n]))
                for w in nb[n]:
                    nxt[idx[w]] += share
            nxt[idx[target]] += 0.15
            pr = nxt
    ppr = {c: pr[idx[c]] for c in cand}

    def top1(sc):
        if not sc:
            return 0
        ts = sc.get(true_ind, 0.0)
        if ts <= 0:
            return 0
        return 1 if all(v < ts for k, v in sc.items() if k != true_ind) else 0

    out = {"CN": top1(cn), "Jac": top1(jac), "AA": top1(aa), "RA": top1(ra),
           "WA": top1(wa), "PPR": top1(ppr), "relpath": top1(rp)}
    out["rand_g"] = 1 if cand and rng.choice(cand) == true_ind else 0
    out["rand_2h"] = 1 if two_hop and rng.choice(two_hop) == true_ind else 0
    return out, len(cand)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", default="15")
    a = ap.parse_args()

    vic = json.load(io.open(os.path.join(
        BASE, "data", "FB15k-237-id-move%s" % a.rate, "victims.json"),
        encoding="utf-8"))
    pat = os.path.join(BASE, "deanon_results",
                       "exp%s_gemini25_batch*.json" % a.rate)
    files = sorted(f for f in glob.glob(pat) if "partial" not in f)
    if not files:
        files = sorted(f for f in glob.glob(os.path.join(
            BASE, "deanon_results", "exp_all_gemini25_batch*.json"))
            if "partial" not in f)

    rng = random.Random(4242)
    names = ["rand_g", "relpath", "rand_2h", "PPR", "agent", "Jac", "CN",
             "WA", "AA", "RA"]
    tally = collections.Counter()
    per_victim = {}
    n = 0
    for f in files:
        d = json.load(io.open(f, encoding="utf-8"))
        for e in d.get("results", []):
            mid = e.get("mid")
            ti = vic.get(mid)
            if not ti:
                continue
            ev = S.evidence_of(e, "final")
            if not ev:
                continue
            res, _ = score_all(ev, mid, ti, rng)
            res["agent"] = 1 if e.get("match") else 0
            n += 1
            for k in names:
                tally[k] += res.get(k, 0)
            per_victim[mid] = res

    print("x=%s%%  n=%d runs, every attacker on the agent's own evidence\n"
          % (a.rate, n))
    for k in names:
        print("   %-8s %4d  %5.2f%%" % (k, tally[k], 100.0 * tally[k] / n))

    os.makedirs(os.path.join(BASE, "deanon_results", "analysis"), exist_ok=True)
    out = os.path.join(BASE, "deanon_results", "analysis",
                       "bm_%s.json" % a.rate)
    json.dump(per_victim, io.open(out, "w", encoding="utf-8"))
    print("\nper-victim written to", out)


if __name__ == "__main__":
    main()
