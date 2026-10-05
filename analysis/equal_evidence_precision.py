"""Precision at coverage on EQUAL EVIDENCE: every index is scored on the final
evidence the agent itself retrieved for that subject (same rules as bm_full.py,
which produced Table V's equal-evidence numbers), and ranked by its own
confidence, the margin between its top two scores (margin 0 = declines).

Agent: answers when it returns a prediction; precision = strict hits
(match AND rank == 1) / answers. Denominator for coverage: all 2,836 subjects;
a subject without stored evidence gives the indices nothing (declines).

Sanity check: index hits must reproduce bm_<rate>.json.
"""
import collections
import glob
import io
import json
import math
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "deanon_results", "analysis")
os.makedirs(RESULTS, exist_ok=True)
HERE = RESULTS
sys.path.insert(0, os.path.join(BASE, "baselines"))
sys.path.insert(0, os.path.join(BASE, "corpus"))
import s3_budget_matched as S

PAT = {"05": "exp_all_gemini25_batch*.json", "10": "exp10_gemini25_batch*.json",
       "15": "exp15_gemini25_batch*.json"}
IDX = ["PPR", "Jac", "CN", "WA", "AA", "RA"]


def scores(triples, target):
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
        return {k: {} for k in IDX}
    n2i = collections.defaultdict(list)
    for c in cand:
        for a in nb[c]:
            n2i[a].append(c)
    anchors = nb.get(target, set())
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
    for r, nn, d in inc.get(target, ()):
        w = idf.get(r, 1.0)
        for c in n2i.get(nn, ()):
            wa[c] += w * aaw.get(nn, 0.0)
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
    return {"CN": cn, "Jac": jac, "AA": aa, "RA": ra, "WA": wa, "PPR": ppr}


def hit_margin(sc, true_ind):
    if not sc:
        return 0, 0.0
    top = sorted(sc.values(), reverse=True)[:2] + [0.0, 0.0]
    margin = top[0] - top[1]
    ts = sc.get(true_ind, 0.0)
    hit = 1 if ts > 0 and all(v < ts for k, v in sc.items() if k != true_ind) else 0
    return hit, margin


def main():
    summary = {}
    for rate in ("05", "10", "15"):
        vic = json.load(io.open(os.path.join(
            BASE, "data", "FB15k-237-id-move%s" % rate, "victims.json"), encoding="utf-8"))
        bm = json.load(io.open(os.path.join(HERE, "bm_%s.json" % rate), encoding="utf-8"))
        N = len(vic)
        rows = {m: {k: (0, 0.0) for k in IDX} for m in vic}
        agent = {m: (False, 0) for m in vic}
        for f in sorted(glob.glob(os.path.join(BASE, "deanon_results", PAT[rate]))):
            if "partial" in f:
                continue
            for e in json.load(io.open(f, encoding="utf-8"))["results"]:
                mid = e.get("mid")
                if mid not in vic:
                    continue
                agent[mid] = (bool(e.get("predictions")),
                              int(bool(e.get("match")) and e.get("rank") == 1))
                ev = S.evidence_of(e, "final")
                if not ev:
                    continue
                sc = scores(ev, mid)
                rows[mid] = {k: hit_margin(sc[k], vic[mid]) for k in IDX}
        check = {k: sum(rows[m][k][0] == bm.get(m, {}).get(k, 0) for m in vic) for k in IDX}

        ans = [m for m in vic if agent[m][0]]
        n_ans = len(ans)
        a_hits = sum(agent[m][1] for m in ans)
        print("\n=== x=%s%%  scorer check vs bm_%s.json: %s" % (
            rate, rate, ", ".join("%s %d/%d" % (k, check[k], N) for k in IDX)))
        print("  agent: answers %d (%.1f%%), %d strict hits, precision %.1f%%"
              % (n_ans, 100.0 * n_ans / N, a_hits, 100.0 * a_hits / n_ans))
        marks = [("1%", round(0.01 * N)), ("5%", round(0.05 * N)),
                 ("10%", round(0.10 * N)), ("agent cov", n_ans)]
        print("  %-4s %6s %6s %8s | %s | %s" % (
            "idx", "hits", "cov", "prec@all", "  ".join("p@%-9s" % k for k, _ in marks),
            "on agent's answered set: idx prec / agent prec"))
        res = {"agent": {"coverage": 100.0 * n_ans / N, "precision": 100.0 * a_hits / n_ans,
                         "hits": a_hits}}
        for k in IDX:
            order = sorted(vic, key=lambda m: -rows[m][k][1])
            pos = [m for m in order if rows[m][k][1] > 0]
            hits = sum(rows[m][k][0] for m in vic)
            cells, rk = [], {}
            for lab, n in marks:
                if 0 < n <= len(pos):
                    p = 100.0 * sum(rows[m][k][0] for m in pos[:n]) / n
                    cells.append("%6.1f%%    " % p)
                    rk[lab] = p
                else:
                    cells.append("    --     ")
            pall = 100.0 * hits / len(pos) if pos else 0.0
            # paired: on the subjects the agent answered, how often is the index right
            # among those where the index also answers?
            both = [m for m in ans if rows[m][k][1] > 0]
            ib = sum(rows[m][k][0] for m in both)
            ab = sum(agent[m][1] for m in both)
            bb = sum(1 for m in both if agent[m][1] and not rows[m][k][0])
            cc = sum(1 for m in both if rows[m][k][0] and not agent[m][1])
            nn = bb + cc
            pm = min(1.0, 2.0 * sum(math.comb(nn, i) for i in range(min(bb, cc) + 1)) / 2.0 ** nn) if nn else 1.0
            # matched coverage: index's top-n_ans by margin vs agent, two-proportion z
            if n_ans <= len(pos):
                ih = sum(rows[m][k][0] for m in pos[:n_ans])
                pp = (ih + a_hits) / (2.0 * n_ans)
                se = math.sqrt(pp * (1 - pp) * 2.0 / n_ans)
                z = (a_hits - ih) / n_ans / se if se else 0.0
                pz = math.erfc(abs(z) / math.sqrt(2))
                mc = "matched-cov p=%.3f" % pz
            else:
                mc = "matched-cov n/a"
            print("     McNemar on paired set: agent-only %d, idx-only %d, p=%.3f; %s" % (bb, cc, pm, mc))
            print("  %-4s %6d %5.1f%% %7.1f%% | %s | n=%d idx %.1f%% / agent %.1f%%" % (
                k, hits, 100.0 * len(pos) / N, pall, "  ".join(cells), len(both),
                100.0 * ib / len(both) if both else 0, 100.0 * ab / len(both) if both else 0))
            res[k] = {"hits": hits, "coverage": 100.0 * len(pos) / N, "prec_all": pall,
                      "at": rk, "paired_n": len(both),
                      "paired_idx": 100.0 * ib / len(both) if both else None,
                      "paired_agent": 100.0 * ab / len(both) if both else None}
        summary[rate] = res

    json.dump(summary, io.open(os.path.join(HERE, "ee_precision.json"), "w",
                               encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
