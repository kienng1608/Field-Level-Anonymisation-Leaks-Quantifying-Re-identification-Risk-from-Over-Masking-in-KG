"""Precision split by reachability: is the true code c_v anywhere in the
evidence the agent saw (cumulative over all rounds)? If not, no attacker
restricted to that evidence can be right, so every answer there is wrong.

Reports, per rate, for the agent and each index:
  - answers and hits in the reachable / unreachable groups
  - precision on reachable answers only
  - matched coverage inside the reachable group (index top-n by margin, n =
    agent's reachable answers) and paired McNemar on subjects both answer.
"""
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equal_evidence_precision import (BASE, PAT, IDX, S, glob, json, scores,  # noqa: E402
                                      hit_margin)

RESULTS = os.path.join(BASE, "deanon_results", "analysis")
os.makedirs(RESULTS, exist_ok=True)
HERE = RESULTS


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2.0 ** n)


def ztest(h1, h2, n):
    p = (h1 + h2) / (2.0 * n)
    se = math.sqrt(p * (1 - p) * 2.0 / n) if 0 < p < 1 else 0
    return math.erfc(abs((h1 - h2) / n / se) / math.sqrt(2)) if se else 1.0


out = {}
for rate in ("05", "10", "15"):
    vic = json.load(io.open(os.path.join(
        BASE, "data", "FB15k-237-id-move%s" % rate, "victims.json"), encoding="utf-8"))
    N = len(vic)
    rows = {m: {k: (0, 0.0) for k in IDX} for m in vic}
    agent = {m: (False, 0) for m in vic}
    reach = {m: False for m in vic}
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
            ti = vic[mid]
            reach[mid] = any(ti == h or ti == t for h, _r, t in ev)
            sc = scores(ev, mid)
            rows[mid] = {k: hit_margin(sc[k], ti) for k in IDX}

    R = [m for m in vic if reach[m]]
    U = [m for m in vic if not reach[m]]
    aR = [m for m in R if agent[m][0]]
    aU = [m for m in U if agent[m][0]]
    hR = sum(agent[m][1] for m in aR)
    hU = sum(agent[m][1] for m in aU)
    print("\n=== x=%s%%: c_v in evidence for %d of %d subjects (%.1f%%)"
          % (rate, len(R), N, 100.0 * len(R) / N))
    print("  agent: answers %d = %d reachable + %d unreachable (%.0f%% of its answers"
          " cannot be right)" % (len(aR) + len(aU), len(aR), len(aU),
                                 100.0 * len(aU) / (len(aR) + len(aU))))
    print("         answers %.1f%% of reachable, %.1f%% of unreachable subjects"
          % (100.0 * len(aR) / len(R), 100.0 * len(aU) / len(U)))
    print("         hits %d (unreachable %d); precision all %.1f%%, on reachable answers %.1f%%"
          % (hR + hU, hU, 100.0 * (hR + hU) / (len(aR) + len(aU)), 100.0 * hR / len(aR)))
    res = {"N": N, "reach": len(R), "agent": {
        "ans_R": len(aR), "ans_U": len(aU), "hits": hR,
        "prec_all": 100.0 * hR / (len(aR) + len(aU)), "prec_R": 100.0 * hR / len(aR)}}
    print("  %-4s %8s %8s %9s %9s | matched cov in R (n=%d): idx / agent, p | paired in R: a-only i-only p"
          % ("idx", "ans_R", "ans_U", "prec_all", "prec_R", len(aR)))
    for k in IDX:
        ansR = [m for m in R if rows[m][k][1] > 0]
        ansU = [m for m in U if rows[m][k][1] > 0]
        hits = sum(rows[m][k][0] for m in R)
        pa = 100.0 * hits / (len(ansR) + len(ansU)) if ansR or ansU else 0
        pr = 100.0 * hits / len(ansR) if ansR else 0
        order = sorted(ansR, key=lambda m: -rows[m][k][1])
        if len(aR) <= len(order):
            ih = sum(rows[m][k][0] for m in order[:len(aR)])
            mc = "%5.1f%% / %5.1f%%, p=%.4f" % (100.0 * ih / len(aR), 100.0 * hR / len(aR),
                                               ztest(hR, ih, len(aR)))
        else:
            mc = "   n/a (idx answers only %d)    " % len(order)
        both = [m for m in aR if rows[m][k][1] > 0]
        b = sum(1 for m in both if agent[m][1] and not rows[m][k][0])
        c = sum(1 for m in both if rows[m][k][0] and not agent[m][1])
        print("  %-4s %8d %8d %8.1f%% %8.1f%% | %s | n=%d %3d %3d p=%.4f"
              % (k, len(ansR), len(ansU), pa, pr, mc, len(both), b, c, mcnemar(b, c)))
        res[k] = {"ans_R": len(ansR), "ans_U": len(ansU), "hits": hits,
                  "prec_all": pa, "prec_R": pr, "matched": mc,
                  "paired": [len(both), b, c, mcnemar(b, c)]}
    out[rate] = res

json.dump(out, io.open(os.path.join(HERE, "ee_reach.json"), "w", encoding="utf-8"), indent=1)
