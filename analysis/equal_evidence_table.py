"""Every attacker on the agent's own evidence (equal evidence), on the same conventions
as Table III:
  - agent strict top-1: match AND rank == 1 (drops the one rank-2 hit at 10%)
  - denominator 2,836 at every rate; a subject whose run stored no evidence
    scores 0 for every index (nothing to score on)
  - McNemar agent vs each of six indices, three rates, Bonferroni over 18
Also checks the random-control coincidence flagged in review.
"""
import glob
import io
import json
import math
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "deanon_results", "analysis")
os.makedirs(RESULTS, exist_ok=True)
HERE = RESULTS
PAT = {"05": "exp_all_gemini25_batch*.json", "10": "exp10_gemini25_batch*.json",
       "15": "exp15_gemini25_batch*.json"}
ORDER = ["rand_g", "relpath", "rand_2h", "PPR", "agent", "Jac", "CN", "WA",
         "AA", "RA"]
IDX = ["PPR", "Jac", "CN", "WA", "AA", "RA"]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)


full = json.load(io.open(os.path.join(
    BASE, "deanon_results", "non_llm_baselines_per_victim.json"),
    encoding="utf-8"))

table = {}
tests = []
for rate in ("05", "10", "15"):
    vic = json.load(io.open(os.path.join(
        BASE, "data", "FB15k-237-id-move%s" % rate, "victims.json"),
        encoding="utf-8"))
    agent = {}
    for f in sorted(glob.glob(os.path.join(BASE, "deanon_results", PAT[rate]))):
        if "partial" in f:
            continue
        for e in json.load(io.open(f, encoding="utf-8"))["results"]:
            agent[e["mid"]] = 1 if (e.get("match") and e.get("rank") == 1) else 0
    bm = json.load(io.open(os.path.join(HERE, "bm_%s.json" % rate),
                           encoding="utf-8"))
    mids = sorted(vic)
    N = len(mids)
    rec = {}
    for m in mids:
        r = dict(bm.get(m, {}))
        r["agent"] = agent.get(m, 0)
        rec[m] = r
    table[rate] = {k: (sum(rec[m].get(k, 0) for m in mids), N) for k in ORDER}
    ag = [rec[m]["agent"] for m in mids]
    for k in IDX:
        o = [rec[m].get(k, 0) for m in mids]
        b = sum(1 for a, x in zip(ag, o) if a and not x)
        c = sum(1 for a, x in zip(ag, o) if x and not a)
        tests.append((rate, k, b, c, mcnemar(b, c)))
    missing = [m for m in mids if m not in bm]
    print("x=%s: N=%d, runs without stored evidence: %d, agent strict hits %d"
          % (rate, N, len(missing), sum(ag)))

print("\n=== Table V (equal evidence), strict top-1, over 2,836 ===")
print("%-9s %8s %8s %8s" % ("", "x=5%", "x=10%", "x=15%"))
for k in ORDER:
    print("%-9s %s" % (k, " ".join("%4d=%5.2f" % (table[r][k][0],
                                                   100.0 * table[r][k][0] /
                                                   table[r][k][1])
                                   for r in ("05", "10", "15"))))

thr = 0.05 / len(tests)
print("\n=== McNemar, agent vs index (Bonferroni over %d, threshold %.5f) ==="
      % (len(tests), thr))
for rate, k, b, c, p in tests:
    lead = "agent" if b > c else "index" if c > b else "tie"
    tag = "SURVIVES" if p < thr else ("nominal" if p < 0.05 else "")
    print("  x=%s %-4s agent-only %3d  idx-only %3d  p=%.4f  %s ahead  %s"
          % (rate, k, b, c, p, lead, tag))

print("\n=== random-control coincidence check at x=5% ===")
f5 = full["rate_05"]
print("  full graph Random(2-hop): %d / %d"
      % (sum(int(v["random_2hop"]) for v in f5.values()), len(f5)))
print("  equal evidence Random(visible codes): %d / 2836" % table["05"]["rand_g"][0])
