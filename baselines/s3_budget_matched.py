"""
S3: does the heuristics' advantage come from the method or from the access?

The heuristics score every candidate in the release; the agent sees a bounded
subgraph it selects itself. A reviewer asked whether the gap is about the
scoring rule or about how much graph each attacker is shown. This answers it
without new API calls: we replay each stored agent run and score Adamic-Adar
and Resource Allocation on exactly the evidence that agent saw.

Evidence comes from the stored run, not from the prompt text:

    rounds[0]["evidence_at_round_start"]   the initial retrieval
    rounds[k]["new_triples_received"]      what each request returned

Both are lists of [head, relation, tail] already written with the target's real
MID. (An earlier version regex-scraped the round-1 prompt instead. That swept in
the triples of the prompt's "=== EXAMPLES ===" block -- illustrative rows such as
Nobel Prize in Physics 2025 -- and, because those examples also use the literal
string "[TARGET PERSON]", four of them were rewritten onto the target node. Every
victim then gained the same three fake anchors carrying low-degree fake
candidates, which Resource Allocation weights most heavily; the heuristics'
scores collapsed as a result. Reading the stored lists avoids the problem
entirely.)

--scope final   every triple the agent ever saw (default). This is the fair
                comparison: the agent chooses what to retrieve over five rounds,
                so limiting the indices to round 1 alone would favour the agent.
--scope round1  the initial allocation only, for reference.

Usage:
  python baselines/s3_budget_matched.py --rate 15
  python baselines/s3_budget_matched.py --rate 15 --scope round1
"""
import argparse
import collections
import glob
import io
import json
import math
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evidence_of(entry, scope):
    """The triples the agent was shown, read from the stored run."""
    rounds = entry.get("rounds") or []
    if not rounds:
        return []
    ev = [tuple(t) for t in (rounds[0].get("evidence_at_round_start") or [])
          if len(t) == 3]
    if scope == "final":
        for r in rounds:
            ev.extend(tuple(t) for t in (r.get("new_triples_received") or [])
                      if len(t) == 3)
    return ev


def score_on(triples, target, true_ind):
    """Adamic-Adar and Resource Allocation over just this subgraph.

    Same anchor rule as the full-graph run: an anchor is any entity both keys
    touch, person codes included. Strict top-1; a tie counts as a failure.
    """
    nb = collections.defaultdict(set)
    for h, r, t in triples:
        nb[h].add(t)
        nb[t].add(h)
    deg = {k: len(v) for k, v in nb.items()}
    cand = [n for n in nb if n.startswith("IND_")]
    n2i = collections.defaultdict(list)
    for c in cand:
        for a in nb[c]:
            n2i[a].append(c)

    out = {}
    for name in ("AA", "RA"):
        w = ({n: 1.0 / math.log(max(2, d)) for n, d in deg.items()} if name == "AA"
             else {n: 1.0 / max(1, d) for n, d in deg.items()})
        sc = collections.defaultdict(float)
        for a in nb.get(target, ()):
            for c in n2i.get(a, ()):
                sc[c] += w.get(a, 0.0)
        ts = sc.get(true_ind, 0.0)
        out[name] = 1 if (ts > 0 and all(v < ts for k, v in sc.items() if k != true_ind)) else 0
    return out, len(cand), (true_ind in nb)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rate", default="15")
    ap.add_argument("--scope", default="final", choices=["final", "round1"])
    args = ap.parse_args()

    vic_path = os.path.join(BASE, "data", "FB15k-237-id-move%s" % args.rate, "victims.json")
    victims = json.load(io.open(vic_path, encoding="utf-8"))

    pattern = os.path.join(BASE, "deanon_results", "exp%s_gemini25_batch*.json" % args.rate)
    files = sorted(f for f in glob.glob(pattern) if "partial" not in f)
    if not files:
        # the x=5% full-corpus run is stored under a different prefix
        pattern = os.path.join(BASE, "deanon_results", "exp_all_gemini25_batch*.json")
        files = sorted(f for f in glob.glob(pattern) if "partial" not in f)
    if not files:
        print("no result files matching", pattern)
        sys.exit(1)

    tally = collections.Counter()
    reach = collections.Counter()
    n = n_reach = 0
    pool_sizes = []
    ev_sizes = []
    for f in files:
        data = json.load(io.open(f, encoding="utf-8"))
        for e in data.get("results", []):
            mid = e.get("mid")
            true_ind = victims.get(mid)
            if not true_ind:
                continue
            ev = evidence_of(e, args.scope)
            if not ev:
                continue
            res, npool, in_ev = score_on(ev, mid, true_ind)
            n += 1
            pool_sizes.append(npool)
            ev_sizes.append(len(ev))
            tally["AA"] += res["AA"]
            tally["RA"] += res["RA"]
            tally["agent"] += 1 if e.get("match") else 0
            if in_ev:
                n_reach += 1
                reach["AA"] += res["AA"]
                reach["RA"] += res["RA"]
                reach["agent"] += 1 if e.get("match") else 0

    med = lambda xs: sorted(xs)[len(xs) // 2]
    print("budget-matched scoring at x=%s%%, scope=%s  (%d agent runs replayed)"
          % (args.rate, args.scope, n))
    print("  evidence per run: median %d triples" % med(ev_sizes))
    print("  candidate codes visible in it: median %d" % med(pool_sizes))
    print()
    for k in ("agent", "AA", "RA"):
        print("  %-6s %4d hits  = %5.2f%%" % (k, tally[k], tally[k] / n * 100))
    print()
    print("  restricted to the %d runs whose identified key is in the evidence:" % n_reach)
    for k in ("agent", "AA", "RA"):
        print("    %-6s %4d hits  = %5.2f%%" % (k, reach[k], reach[k] / n_reach * 100))


if __name__ == "__main__":
    main()
