"""
Extract every number the paper needs from the 3.1 GB of raw result files into a
handful of small CSV/JSON files under paper_handoff/.

The raw exp_*.json files each carry the full multi-round conversation transcript
for every target, which is what makes them ~100 MB apiece. A paper needs only the
aggregates and a few illustrative transcripts, so this script reduces them to
files small enough to hand to a writing assistant along with the source code.

Outputs (all under paper_handoff/):
  data/per_victim.csv          one row per (victim, leak rate): L, k, hit, relations
  data/dose_response.csv       P(hit | L) pooled across both leak rates
  data/anonymity_strata.csv    hit rate by anonymity-set size bucket
  data/per_relation.csv        hit rate by sensitive relation
  data/model_fit.json          the three fitted models, their parameters and AIC
  data/corpus_stats.json       graph/corpus statistics quoted in the paper
  cases/*.txt                  full transcripts of the illustrative cases

Usage:
  python codes/extract_paper_data.py
"""
import collections
import csv
import io
import json
import math
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper_handoff")
os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
os.makedirs(os.path.join(OUT, "cases"), exist_ok=True)

SENSITIVE_POSITION = {
    "/medicine/disease/notable_people_with_this_condition": "tail",
    "/people/cause_of_death/people": "tail",
    "/celebrities/celebrity/sexual_relationships./celebrities/romantic_relationship/celebrity": "both",
    "/people/person/religion": "head",
    "/people/ethnicity/people": "tail",
    "/government/political_party/politicians_in_this_party./government/political_party_tenure/politician": "tail",
    "/base/schemastaging/person_extra/net_worth./measurement_unit/dated_money_value/currency": "head",
}

RUNS = [
    # (label, leak rate, leaked-graph dir, result-file prefix)
    # A run is included only if its result files exist, so this list can name
    # runs that have not finished yet.
    ("5pct", 0.05, "FB15k-237-id-move05", "exp_all_gemini25_batch"),
    ("10pct", 0.10, "FB15k-237-id-move10", "exp10_gemini25_batch"),
    ("15pct", 0.15, "FB15k-237-id-move15", "exp15_gemini25_batch"),
    ("20pct", 0.20, "FB15k-237-id-move20", "exp20_gemini25_batch"),
]


def run_is_available(graph_dir, prefix):
    """True when the leaked graph and at least one result batch are on disk."""
    if not os.path.exists(os.path.join(ROOT, "data", graph_dir, "train.txt")):
        return False
    return any(
        os.path.exists(os.path.join(ROOT, "deanon_results", f"{prefix}{i}{suffix}"))
        for i in range(1, 7) for suffix in (".json", ".partial.json")
    )


def load_triples(path):
    triples = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples


def leaked_edge_counts(base_triples, leaked_path, victim_mids):
    """How many edges were MOVEd onto each victim's MID key (their L)."""
    leaked = load_triples(leaked_path)
    counts = collections.Counter()
    for before, after in zip(base_triples, leaked):
        if before == after:
            continue
        for endpoint in (after[0], after[2]):
            if endpoint in victim_mids:
                counts[endpoint] += 1
    return counts


def hits_from(prefix):
    """MIDs the attack resolved correctly, across all six batches of one run."""
    hits = set()
    n_targets = 0
    for i in range(1, 7):
        for suffix in (".json", ".partial.json"):
            path = os.path.join(ROOT, "deanon_results", f"{prefix}{i}{suffix}")
            if not os.path.exists(path):
                continue
            with io.open(path, encoding="utf-8") as f:
                data = json.load(f)
            for r in data.get("results", []):
                n_targets += 1
                if r.get("match"):
                    hits.add(r["mid"])
            break
    return hits, n_targets


def sensitive_profile(triples, victim_mids):
    """Per victim: which sensitive relations they carry, and the anonymity-set
    size of each sensitive VALUE (how many victims share that exact value)."""
    value_to_victims = collections.defaultdict(set)
    mid_to_values = collections.defaultdict(list)
    mid_to_relations = collections.defaultdict(set)
    for h, r, t in triples:
        pos = SENSITIVE_POSITION.get(r)
        if pos is None:
            continue
        if pos in ("tail", "both") and t in victim_mids:
            value_to_victims[(r, h)].add(t)
            mid_to_values[t].append((r, h))
            mid_to_relations[t].add(r)
        if pos in ("head", "both") and h in victim_mids:
            value_to_victims[(r, t)].add(h)
            mid_to_values[h].append((r, t))
            mid_to_relations[h].add(r)
    # each victim's SMALLEST anonymity set = their most identifying attribute
    mid_min_k = {
        m: min(len(value_to_victims[v]) for v in vals)
        for m, vals in mid_to_values.items() if vals
    }
    return value_to_victims, mid_min_k, mid_to_relations


def main():
    print("Loading base (pre-leak) graph ...")
    base_dir = os.path.join(ROOT, "data", "FB15k-237-id-masked")
    base_triples = load_triples(os.path.join(base_dir, "train.txt"))
    victims = json.load(io.open(os.path.join(base_dir, "victims.json"), encoding="utf-8"))
    victim_mids = set(victims)
    print(f"  {len(base_triples):,} triples, {len(victim_mids):,} victims")

    value_to_victims, mid_min_k, mid_to_rels = sensitive_profile(base_triples, victim_mids)

    # Victim degree on the identified key, counting non-sensitive edges only.
    # L is a fraction of this, so the two are strongly correlated and any model
    # of L has to be able to control for it.
    ind_degree = collections.Counter()
    for h, r, t in base_triples:
        if r in SENSITIVE_POSITION:
            continue
        for endpoint in (h, t):
            if endpoint.startswith("IND_"):
                ind_degree[endpoint] += 1
    mid_degree = {mid: ind_degree.get(code, 0) for mid, code in victims.items()}

    runs = [r for r in RUNS if run_is_available(r[2], r[3])]
    skipped = [r[0] for r in RUNS if r not in runs]
    print(f"  runs found: {', '.join(r[0] for r in runs)}"
          + (f"   (skipped, no results yet: {', '.join(skipped)})" if skipped else ""))

    # ---- per-victim rows, one per (victim, leak rate) ----
    rows = []
    per_run = {}
    for label, rate, graph_dir, prefix in runs:
        leaked_path = os.path.join(ROOT, "data", graph_dir, "train.txt")
        L_of = leaked_edge_counts(base_triples, leaked_path, victim_mids)
        hits, n_targets = hits_from(prefix)
        per_run[label] = {
            "leak_rate": rate,
            "n_targets_run": n_targets,
            "n_hits": len(hits),
            "hit_rate": len(hits) / len(victim_mids),
        }
        for mid in sorted(victim_mids):
            rows.append({
                "mid": mid,
                "leak_rate": rate,
                "L": L_of.get(mid, 0),
                "anonymity_set_k": mid_min_k.get(mid, ""),
                "n_sensitive_relations": len(mid_to_rels.get(mid, ())),
                "hit": 1 if mid in hits else 0,
                "degree": mid_degree.get(mid, 0),
            })
        print(f"  {label}: {len(hits)} hits / {len(victim_mids)} victims")

    with io.open(os.path.join(OUT, "data", "per_victim.csv"), "w",
                 encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> per_victim.csv ({len(rows)} rows)")

    # ---- dose-response: P(hit | L), pooled ----
    by_L = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        by_L[r["L"]][0] += 1
        by_L[r["L"]][1] += r["hit"]
    with io.open(os.path.join(OUT, "data", "dose_response.csv"), "w",
                 encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["L", "n_observations", "n_hits", "p_hat"])
        for L in sorted(by_L):
            n, h = by_L[L]
            w.writerow([L, n, h, f"{h/n:.6f}"])
    print("  -> dose_response.csv")

    # ---- model fitting ----
    items = sorted(by_L.items())

    def nll(fn):
        s = 0.0
        for L, (n, h) in items:
            p = min(max(fn(L), 1e-12), 1 - 1e-12)
            s -= h * math.log(p) + (n - h) * math.log(1 - p)
        return s

    best_indep = min(
        ((nll(lambda L, p=i/100000: 1 - (1 - p)**L), i/100000) for i in range(1, 3000)),
        key=lambda x: x[0])
    best_sat = min(
        ((nll(lambda L, A=a/1000, tau=t/100: A * (1 - math.exp(-L / tau))), a/1000, t/100)
         for a in range(1, 400) for t in range(1, 400)),
        key=lambda x: x[0])
    best_log = min(
        ((nll(lambda L, a=ai/100, b=bi/100: 1/(1 + math.exp(-(a + b*L)))), ai/100, bi/100)
         for ai in range(-800, 0, 2) for bi in range(1, 120)),
        key=lambda x: x[0])

    fit = {
        "n_observations": sum(n for _, (n, _) in items),
        "n_hits": sum(h for _, (_, h) in items),
        "models": {
            "independence": {"form": "P = 1-(1-p)^L", "p": best_indep[1],
                             "nll": best_indep[0], "k_params": 1,
                             "aic": 2*best_indep[0] + 2},
            "saturating": {"form": "P = A*(1-exp(-L/tau))", "A": best_sat[1],
                           "tau": best_sat[2], "nll": best_sat[0], "k_params": 2,
                           "aic": 2*best_sat[0] + 4},
            "logistic": {"form": "P = sigmoid(a+b*L)", "a": best_log[1], "b": best_log[2],
                         "nll": best_log[0], "k_params": 2, "aic": 2*best_log[0] + 4},
        },
        "selected": "saturating",
    }

    # goodness of fit for the selected model
    A, tau = best_sat[1], best_sat[2]
    chi2, dof = 0.0, 0
    for L, (n, h) in items:
        if n < 50:
            continue
        p = A * (1 - math.exp(-L / tau))
        e = n * p
        if e >= 5:
            chi2 += (h - e)**2 / e + ((n - h) - (n - e))**2 / (n - e)
            dof += 1
    fit["goodness_of_fit"] = {"pearson_chi2": chi2, "approx_dof": max(dof - 2, 1)}

    # profile-likelihood 95% CI (chi2 1df cutoff = 1.92)
    base_nll = best_sat[0]

    def nll_sat(A, tau):
        return nll(lambda L, A=A, tau=tau: A * (1 - math.exp(-L / tau)))

    A_ok = [a/1000 for a in range(20, 300)
            if min(nll_sat(a/1000, t/100) for t in range(50, 1200, 10)) - base_nll <= 1.92]
    tau_ok = [t/100 for t in range(100, 1500, 5)
              if min(nll_sat(a/1000, t/100) for a in range(20, 300, 2)) - base_nll <= 1.92]
    fit["ci95"] = {
        "A": [min(A_ok), max(A_ok)] if A_ok else None,
        "tau": [min(tau_ok), max(tau_ok)] if tau_ok else None,
    }

    json.dump(fit, io.open(os.path.join(OUT, "data", "model_fit.json"), "w",
                           encoding="utf-8"), indent=2)
    print(f"  -> model_fit.json (selected: saturating, A={A:.4f}, tau={tau:.2f})")

    # ---- anonymity-set strata ----
    buckets = [(1, 1, "k=1"), (2, 4, "k=2-4"), (5, 19, "k=5-19"),
               (20, 99, "k=20-99"), (100, 10**9, "k>=100")]
    with io.open(os.path.join(OUT, "data", "anonymity_strata.csv"), "w",
                 encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "n_victims", "hit_rate_5pct", "hit_rate_10pct"])
        for lo, hi, label in buckets:
            grp = [m for m, k in mid_min_k.items() if lo <= k <= hi]
            out = [label, len(grp)]
            for run_label, rate, graph_dir, prefix in runs:
                hits, _ = hits_from(prefix)
                out.append(f"{sum(1 for m in grp if m in hits)/len(grp):.6f}" if grp else "")
            w.writerow(out)
    print("  -> anonymity_strata.csv")

    # ---- per-relation ----
    with io.open(os.path.join(OUT, "data", "per_relation.csv"), "w",
                 encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["relation", "n_victims", "hit_rate_5pct", "hit_rate_10pct"])
        for rel in SENSITIVE_POSITION:
            grp = [m for m in victim_mids if rel in mid_to_rels.get(m, ())]
            if not grp:
                continue
            out = [rel, len(grp)]
            for run_label, rate, graph_dir, prefix in runs:
                hits, _ = hits_from(prefix)
                out.append(f"{sum(1 for m in grp if m in hits)/len(grp):.6f}")
            w.writerow(out)
    print("  -> per_relation.csv")

    # ---- corpus statistics ----
    anon_sizes = sorted(len(v) for v in value_to_victims.values())
    n_as = len(anon_sizes)
    degree = collections.Counter()
    for h, r, t in base_triples:
        if r in SENSITIVE_POSITION:
            continue
        for e in (h, t):
            if e.startswith("IND_"):
                degree[e] += 1
    victim_degrees = sorted(degree.get(victims[m], 0) for m in victim_mids)
    nd = len(victim_degrees)
    largest = sorted(value_to_victims.items(), key=lambda x: -len(x[1]))[:10]

    stats = {
        "n_triples": len(base_triples),
        "n_victims": len(victim_mids),
        "runs": per_run,
        "anonymity_set_sizes": {
            "n_distinct_values": n_as,
            "min": anon_sizes[0], "p25": anon_sizes[n_as//4],
            "median": anon_sizes[n_as//2], "p75": anon_sizes[3*n_as//4],
            "max": anon_sizes[-1],
        },
        "largest_anonymity_sets": [
            {"value": v, "relation": r, "n_victims": len(vs)} for (r, v), vs in largest
        ],
        "victim_nonsensitive_degree": {
            "min": victim_degrees[0], "p25": victim_degrees[nd//4],
            "median": victim_degrees[nd//2], "p75": victim_degrees[3*nd//4],
            "p95": victim_degrees[int(nd*0.95)], "max": victim_degrees[-1],
        },
    }
    for label, rate, graph_dir, prefix in runs:
        leaked_path = os.path.join(ROOT, "data", graph_dir, "train.txt")
        L_of = leaked_edge_counts(base_triples, leaked_path, victim_mids)
        vals = sorted(L_of.values())
        nv = len(vals)
        stats["runs"][label]["victims_with_zero_leak"] = len(victim_mids) - nv
        stats["runs"][label]["leaked_edges_per_victim"] = {
            "min": vals[0], "median": vals[nv//2], "max": vals[-1],
            "mean": sum(vals)/nv,
        }
    json.dump(stats, io.open(os.path.join(OUT, "data", "corpus_stats.json"), "w",
                             encoding="utf-8"), indent=2, ensure_ascii=False)
    print("  -> corpus_stats.json")

    # ---- illustrative transcripts ----
    wanted = {
        "/m/EXAMPLE01": ("exp_all_gemini25_batch",
                     "clean_derivation_recognition_confessed_in_step_E"),
        "/m/EXAMPLE02": ("exp10_gemini25_batch",
                     "single_specific_anchor_suffices"),
        "/m/EXAMPLE03": ("exp_all_gemini25_batch",
                     "award_intersection_isolates_one_code"),
    }
    found = set()
    for mid, (prefix, name) in wanted.items():
      for i in range(1, 7):
        if mid in found:
            break
        for suffix in (".json", ".partial.json"):
            path = os.path.join(ROOT, "deanon_results", f"{prefix}{i}{suffix}")
            if not os.path.exists(path):
                continue
            with io.open(path, encoding="utf-8") as f:
                data = json.load(f)
            for r in data.get("results", []):
                if r["mid"] == mid and mid not in found and r.get("match"):
                    msgs = [m["content"] for m in r.get("conversation", [])
                            if m["role"] == "assistant" and m["content"].strip()]
                    with io.open(os.path.join(OUT, "cases", f"{name}.txt"), "w",
                                 encoding="utf-8") as cf:
                        cf.write(f"MID: {r['mid']}\nGround truth: {r['real_name']}\n")
                        cf.write(f"Rounds used: {r.get('rounds_used')}\n")
                        cf.write("=" * 70 + "\n\nFINAL DERIVATION:\n\n")
                        cf.write(msgs[-1] if msgs else "(empty)")
                    found.add(mid)
            break
    print(f"  -> cases/ ({len(found)} transcripts)")

    print(f"\nDone. Everything the paper needs is under {OUT}")


if __name__ == "__main__":
    main()
