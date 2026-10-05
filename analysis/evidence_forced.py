"""Does the evidence force the answer? Three deterministic rules, applied to
every hit, recomputed from the stored evidence without consulting the model.

  count  : shared-anchor count (the rule of the supplement's worked hit)
  rare   : the same anchors weighted by 1/log deg(a) inside the evidence
           (Adamic-Adar restricted to what the agent saw)
  either : the true code is the unique maximiser under at least one of them

A hit is "forced" when the true code is the unique maximiser: any procedure
applying that rule to those triples returns it, whatever the model knew.
"""
import collections
import glob
import io
import json
import math
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "baselines"))
import s3_budget_matched as S  # noqa: E402

PAT = {"05": "exp_all_gemini25_batch*.json", "10": "exp10_gemini25_batch*.json",
       "15": "exp15_gemini25_batch*.json"}


def verdicts(ev, mid, true):
    nb = collections.defaultdict(set)
    for h, r, t in ev:
        nb[h].add(t)
        nb[t].add(h)
    anchors = nb.get(mid, set())
    count = collections.Counter()
    rare = collections.defaultdict(float)
    for a in anchors:
        w = 1.0 / math.log(max(2, len(nb.get(a, ()))))
        for c in nb.get(a, ()):
            if c.startswith("IND_") and c != mid:
                count[c] += 1
                rare[c] += w
    out = {}
    for name, sc in (("count", count), ("rare", rare)):
        if not sc or true not in sc:
            out[name] = "no candidate"
            continue
        best = max(sc.values())
        if sc[true] < best:
            out[name] = "not forced"
        elif sum(1 for v in sc.values() if v == best) == 1:
            out[name] = "forced"
        else:
            out[name] = "tied"
    return out


rows = {}
for rate in ("05", "10", "15"):
    vic = json.load(io.open(os.path.join(
        BASE, "data", "FB15k-237-id-move%s" % rate, "victims.json"), encoding="utf-8"))
    t = collections.Counter()
    n = 0
    for f in sorted(x for x in glob.glob(os.path.join(BASE, "deanon_results", PAT[rate]))
                    if "partial" not in x):
        for e in json.load(io.open(f, encoding="utf-8"))["results"]:
            if not (e.get("match") and e.get("rank") == 1):
                continue
            n += 1
            ev = S.evidence_of(e, "final")
            if not ev:
                t["no evidence"] += 1
                continue
            v = verdicts(ev, e["mid"], vic[e["mid"]])
            t["count " + v["count"]] += 1
            t["rare " + v["rare"]] += 1
            if "forced" in (v["count"], v["rare"]):
                t["either forced"] += 1
    rows[rate] = (n, dict(t))
    print("x=%s%%  %d hits:  count-forced %d (%.0f%%)   rarity-forced %d (%.0f%%)   "
          "either %d (%.0f%%)" % (rate, n, t["count forced"], 100.0 * t["count forced"] / n,
                                  t["rare forced"], 100.0 * t["rare forced"] / n,
                                  t["either forced"], 100.0 * t["either forced"] / n))
tot = sum(v[0] for v in rows.values())
eith = sum(v[1].get("either forced", 0) for v in rows.values())
rare = sum(v[1].get("rare forced", 0) for v in rows.values())
print("\nall rates: %d of %d hits (%.1f%%) are forced by rarity weighting; "
      "%d (%.1f%%) by at least one rule" % (rare, tot, 100.0 * rare / tot, eith, 100.0 * eith / tot))
RESULTS = os.path.join(BASE, "deanon_results", "analysis")
os.makedirs(RESULTS, exist_ok=True)
json.dump(rows, io.open(os.path.join(RESULTS, "evidence_forced.json"), "w",
                        encoding="utf-8"), indent=1)
