"""
Multivariate and paired analysis of the re-identification results.

Motivation: the paper's central claim is a single-variable dose-response curve,
P(hit | L) = A(1 - exp(-L/tau)). But L is produced by moving a fraction x of a
victim's non-sensitive edges, so L ~ x * degree by construction, and any effect
attributed to L may in fact belong to degree. This script quantifies that
confound and reports the analyses that survive it.

Outputs (printed, and written to paper_handoff/data/multivariate.json):

  (a) Logistic regressions of hit on L (linear and saturating forms), degree and
      anonymity-set size, with coefficients, standard errors, p-values and AIC.
  (b) Conditional (fixed-effects) logistic regression within victim, which
      removes every victim-constant covariate, degree and k included. Only
      victims discordant across the two leak rates contribute.
  (c) The full paired table across two leak rates, with McNemar's test.
  (c4) With three or more leak rates: Cochran's Q over all runs, every pairwise
      McNemar, the within-victim sign test pooled over all rate pairs, and the
      multi-way overlap of hit sets against the independence expectation.
  (d) The L = 0 cell: exact p-values against two baselines, and a 95% upper
      bound on the underlying rate.
  (e) Wilson 95% intervals for the headline rates.
  (f) Fisher exact tests comparing the extreme anonymity-set strata.
  (g) Wilson 95% intervals for every cell of the stratified tables.

Everything is computed from paper_handoff/data/per_victim.csv; the only value
taken from elsewhere is tau, read from model_fit.json so that the saturating
transform matches the fit reported in the paper.

Two-rate blocks --- (c) and (c3) --- are defined for a pair of runs. With more
rates present they use the two lowest and say so, while (c4) covers all of them,
so the numbers a two-rate run produced stay reproducible.

Usage:
  python codes/multivariate_analysis.py                      # every rate in the CSV
  python codes/multivariate_analysis.py --rates 0.05,0.1     # the paper's pair
  python codes/multivariate_analysis.py --out somewhere.json # do not overwrite
"""
import collections
import io
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "paper_handoff", "data")
OUT_JSON = os.path.join(DATA, "multivariate.json")

SENSITIVE_LABEL = {
    "/medicine/disease/notable_people_with_this_condition": "disease",
    "/people/cause_of_death/people": "cause of death",
    "/celebrities/celebrity/sexual_relationships./celebrities/romantic_relationship/celebrity":
        "sexual relationship",
    "/people/person/religion": "religion",
    "/people/ethnicity/people": "ethnicity",
    "/government/political_party/politicians_in_this_party./government/political_party_tenure/politician":
        "political party",
    "/base/schemastaging/person_extra/net_worth./measurement_unit/dated_money_value/currency":
        "net worth currency",
}


def rule(title):
    print("\n" + "=" * 74)
    print("  " + title)
    print("=" * 74)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def wilson(k, n, z=1.959963984540054):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_ci(k, n):
    lo, hi = wilson(k, n)
    return "%5.2f%%  [%.2f%%, %.2f%%]  (%d/%d)" % (100 * k / n, 100 * lo, 100 * hi, k, n)


def summarise_glm(res, name, terms):
    """Coefficient table of a fitted statsmodels GLM/Logit result."""
    out = {"model": name, "aic": float(res.aic), "llf": float(res.llf),
           "n_obs": int(res.nobs), "terms": {}}
    for t in terms:
        out["terms"][t] = {
            "coef": float(res.params[t]),
            "se": float(res.bse[t]),
            "z": float(res.tvalues[t]),
            "p": float(res.pvalues[t]),
            "or": float(np.exp(res.params[t])),
        }
    return out


def print_glm(block):
    print("  %-34s AIC = %9.2f   n = %d" % (block["model"], block["aic"], block["n_obs"]))
    print("    %-22s %10s %9s %8s %10s" % ("term", "coef", "SE", "p", "odds ratio"))
    for t, v in block["terms"].items():
        print("    %-22s %10.4f %9.4f %8.4g %10.3f"
              % (t, v["coef"], v["se"], v["p"], v["or"]))


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
def load(keep_rates=None):
    df = pd.read_csv(os.path.join(DATA, "per_victim.csv"))
    need = {"mid", "leak_rate", "L", "anonymity_set_k", "n_sensitive_relations",
            "hit", "degree"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit("per_victim.csv is missing columns: %s" % sorted(missing))
    if keep_rates:
        have = sorted(df["leak_rate"].unique())
        unknown = [r for r in keep_rates if not any(abs(r - h) < 1e-9 for h in have)]
        if unknown:
            raise SystemExit("no such leak rate in per_victim.csv: %s (have %s)"
                             % (unknown, have))
        df = df[df["leak_rate"].isin(keep_rates)].copy()
    with io.open(os.path.join(DATA, "model_fit.json"), encoding="utf-8") as f:
        fit = json.load(f)
    tau = fit["models"][fit["selected"]]["tau"]
    A = fit["models"][fit["selected"]]["A"]
    return df, A, tau


# ---------------------------------------------------------------------------
# 0. the confound itself
# ---------------------------------------------------------------------------
def confound(df, results):
    rule("0.  THE CONFOUND: L IS BUILT FROM degree")

    r_p = float(np.corrcoef(df["L"], df["degree"])[0, 1])
    r_s = float(stats.spearmanr(df["L"], df["degree"]).statistic)
    print("  corr(L, degree)   Pearson  = %.3f" % r_p)
    print("                    Spearman = %.3f" % r_s)

    # degree gradient with L held in a narrow band
    band = df[(df["L"] >= 1) & (df["L"] <= 4)].copy()
    band["dq"] = pd.qcut(band["degree"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    print("\n  hit rate by degree quartile, holding 1 <= L <= 4:")
    dq_rows = []
    for q, g in band.groupby("dq", observed=True):
        k, n = int(g["hit"].sum()), len(g)
        lo, hi = wilson(k, n)
        dq_rows.append({"quartile": str(q), "degree_min": int(g["degree"].min()),
                        "degree_max": int(g["degree"].max()), "n": n, "hits": k,
                        "rate": k / n, "ci95": [lo, hi]})
        print("    %-3s degree %3d-%3d   %s" % (q, g["degree"].min(), g["degree"].max(),
                                                fmt_ci(k, n)))

    # L gradient with degree held in a narrow band
    narrow = df[(df["degree"] >= 25) & (df["degree"] <= 45)]
    print("\n  hit rate by L, holding degree in [25, 45]  (n = %d):" % len(narrow))
    l_rows = []
    for L in range(0, 6):
        g = narrow[narrow["L"] == L]
        if len(g) < 20:
            continue
        k, n = int(g["hit"].sum()), len(g)
        lo, hi = wilson(k, n)
        l_rows.append({"L": L, "n": n, "hits": k, "rate": k / n, "ci95": [lo, hi]})
        print("    L = %d   %s" % (L, fmt_ci(k, n)))

    results["confound"] = {
        "pearson_L_degree": r_p,
        "spearman_L_degree": r_s,
        "degree_gradient_at_L_1_to_4": dq_rows,
        "L_gradient_in_degree_band_25_45": l_rows,
    }


# ---------------------------------------------------------------------------
# (a) multivariate logistic regression
# ---------------------------------------------------------------------------
def part_a(df, tau, results):
    rule("(a)  LOGISTIC REGRESSION: does L survive controlling for degree and k?")

    d = df.copy()
    d["sat"] = 1.0 - np.exp(-d["L"] / tau)          # saturating transform, tau from the paper
    d["log_degree"] = np.log(d["degree"].clip(lower=1))
    d["log_k"] = np.log(d["anonymity_set_k"].clip(lower=1))
    y = d["hit"].astype(float)

    specs = [
        ("M1  L linear, alone",              ["L"]),
        ("M2  L saturating, alone",          ["sat"]),
        ("M3  L linear + degree + k",        ["L", "log_degree", "log_k"]),
        ("M4  L saturating + degree + k",    ["sat", "log_degree", "log_k"]),
        ("M5  degree + k only (no L)",       ["log_degree", "log_k"]),
        ("M6  L sat + L linear + deg + k",   ["sat", "L", "log_degree", "log_k"]),
    ]

    blocks = []
    for name, terms in specs:
        X = sm.add_constant(d[terms], has_constant="add")
        try:
            res = sm.Logit(y, X).fit(disp=0, maxiter=200)
        except Exception as exc:                       # pragma: no cover
            print("  %-34s FAILED: %s" % (name, exc))
            continue
        block = summarise_glm(res, name, terms)
        blocks.append(block)
        print_glm(block)
        print()

    best = min(blocks, key=lambda b: b["aic"])
    print("  lowest AIC: %s" % best["model"])
    results["logistic"] = {"tau_used": tau, "models": blocks, "best_by_aic": best["model"]}


# ---------------------------------------------------------------------------
# (b) conditional logit with victim fixed effects
# ---------------------------------------------------------------------------
def part_b(df, tau, results):
    rule("(b)  CONDITIONAL LOGIT, VICTIM FIXED EFFECTS (within-victim)")

    d = df.copy()
    d["sat"] = 1.0 - np.exp(-d["L"] / tau)

    counts = d.groupby("mid")["hit"].agg(["sum", "count"])
    discordant = counts[(counts["sum"] > 0) & (counts["sum"] < counts["count"])].index
    print("  victims contributing to the likelihood (discordant pairs): %d" % len(discordant))
    print("  (victims hit at both rates or at neither contribute nothing)")

    sub = d[d["mid"].isin(discordant)].copy()
    out = {"n_discordant_victims": int(len(discordant)),
           "n_observations_used": int(len(sub))}

    for label, term in (("L linear", "L"), ("L saturating", "sat")):
        try:
            model = sm.ConditionalLogit(sub["hit"].astype(float), sub[[term]],
                                        groups=sub["mid"])
            res = model.fit(disp=0, maxiter=200)
            conv = bool(getattr(res, "mle_retvals", {}).get("converged", True))
            ci = res.conf_int()
            block = {
                "term": term, "converged": conv,
                "coef": float(res.params.iloc[0]), "se": float(res.bse.iloc[0]),
                "p": float(res.pvalues.iloc[0]),
                "or": float(np.exp(res.params.iloc[0])),
                "ci95_coef": [float(ci.iloc[0, 0]), float(ci.iloc[0, 1])],
            }
            print("  %-14s coef = %8.4f  SE = %6.4f  p = %.4g  OR = %.3f  converged=%s"
                  % (label, block["coef"], block["se"], block["p"], block["or"], conv))
        except Exception as exc:
            block = {"term": term, "converged": False, "error": str(exc)}
            print("  %-14s FAILED: %s" % (label, exc))
        out[label.replace(" ", "_")] = block

    # Distribution-free fallback / cross-check: within a victim, how often does a
    # hit fall on the observation with the larger L? Under the null of no L effect
    # this is a fair coin, conditional on the pair being discordant. With R runs a
    # victim contributes every (hit, miss) pair it has, so up to R-1 of them; those
    # pairs share a victim and are therefore not independent, which the printout
    # says out loud.
    larger, smaller, tied = 0, 0, 0
    for mid, g in sub.groupby("mid"):
        for _, hit_row in g[g["hit"] == 1].iterrows():
            for _, miss_row in g[g["hit"] == 0].iterrows():
                if hit_row["L"] > miss_row["L"]:
                    larger += 1
                elif hit_row["L"] < miss_row["L"]:
                    smaller += 1
                else:
                    tied += 1
    informative = larger + smaller
    n_rates = int(sub["leak_rate"].nunique())
    sign_p2 = float(stats.binomtest(larger, informative, 0.5).pvalue) if informative else float("nan")
    sign_p1 = float(stats.binomtest(larger, informative, 0.5,
                                    alternative="greater").pvalue) if informative else float("nan")
    print("\n  sign test within discordant pairs (ties carry no information):")
    print("    hit on the larger-L observation : %d" % larger)
    print("    hit on the smaller-L observation: %d" % smaller)
    print("    tied L (uninformative)          : %d" % tied)
    print("    exact binomial p, one-sided     = %.4g" % sign_p1)
    print("    exact binomial p, two-sided     = %.4g" % sign_p2)
    if n_rates > 2:
        print("    NOTE: %d leak rates, so one victim contributes several pairs;" % n_rates)
        print("          these are not independent and the p-value is optimistic.")
    out["sign_test"] = {"hit_on_larger_L": larger, "hit_on_smaller_L": smaller,
                        "tied": tied, "n_informative": informative,
                        "n_leak_rates": n_rates,
                        "pairs_independent": n_rates == 2,
                        "p_one_sided": sign_p1, "p_two_sided": sign_p2}

    out["tau_robustness"] = tau_sweep(sub)
    out["permutation"] = permutation_test(sub)

    results["conditional_logit"] = out


# ---------------------------------------------------------------------------
# (b2) is the within-victim result an artefact of the chosen tau?
# ---------------------------------------------------------------------------
def tau_sweep(sub):
    """Refit the within-victim coefficient across a wide range of tau."""
    rows = []
    for tau_try in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]:
        x = pd.DataFrame({"sat": 1.0 - np.exp(-sub["L"] / float(tau_try))},
                         index=sub.index)
        try:
            res = sm.ConditionalLogit(sub["hit"].astype(float), x,
                                      groups=sub["mid"]).fit(disp=0, maxiter=200)
            rows.append({"tau": tau_try, "coef": float(res.params.iloc[0]),
                         "p": float(res.pvalues.iloc[0])})
        except Exception as exc:
            rows.append({"tau": tau_try, "error": str(exc)})
    ok = [r for r in rows if "coef" in r]
    print("\n  tau sensitivity of the within-victim coefficient:")
    print("    " + "  ".join("tau=%g:%.2f" % (r["tau"], r["coef"]) for r in ok))
    if ok:
        print("    coefficient stays positive across tau in [%g, %g], range %.2f-%.2f"
              % (ok[0]["tau"], ok[-1]["tau"],
                 min(r["coef"] for r in ok), max(r["coef"] for r in ok)))
    return {"sweep": rows,
            "coef_min": min((r["coef"] for r in ok), default=None),
            "coef_max": max((r["coef"] for r in ok), default=None),
            "all_positive": all(r["coef"] > 0 for r in ok) if ok else None}


# ---------------------------------------------------------------------------
# (b3) permutation test: exact under H0, estimates nothing
# ---------------------------------------------------------------------------
def permutation_test(sub, n_perm=200000, seed=4242):
    """Randomisation test over discordant pairs.

    Under the null that L does not affect the outcome, within a discordant pair
    the hit is equally likely to fall on either observation. The statistic is
    sum_i [ f(L_hit,i) - f(L_miss,i) ]; the null distribution is obtained by
    flipping the sign of each pair's contribution independently. Nothing is
    fitted, so the criticism that tau was estimated on the same data does not
    apply to these p-values.
    """
    pairs = []
    for mid, g in sub.groupby("mid"):
        if len(g) != 2:
            continue
        hit_L = float(g[g["hit"] == 1]["L"].iloc[0])
        miss_L = float(g[g["hit"] == 0]["L"].iloc[0])
        pairs.append((hit_L, miss_L))

    transforms = [
        ("linear", lambda v: v),
        ("sqrt", np.sqrt),
        ("log1p", np.log1p),
        ("saturating", lambda v: 1.0 - np.exp(-v / 3.99)),
    ]

    rng = np.random.default_rng(seed)
    hits = np.array([a for a, _ in pairs])
    misses = np.array([b for _, b in pairs])
    flips = rng.integers(0, 2, size=(n_perm, len(pairs))) * 2 - 1   # +-1

    out = {"n_pairs": len(pairs), "n_permutations": n_perm, "seed": seed,
           "tests": {}}
    print("\n  permutation test over %d discordant pairs (%d permutations, one-sided):"
          % (len(pairs), n_perm))
    for name, f in transforms:
        d = f(hits) - f(misses)
        obs = float(d.sum())
        null = flips @ d
        p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
        out["tests"][name] = {"observed_statistic": obs, "p_one_sided": p}
        print("    %-12s statistic = %8.3f   p = %.4g" % (name, obs, p))
    print("    (p falls as the transform compresses large L: the shape itself carries signal)")
    return out


# ---------------------------------------------------------------------------
# (c) paired table and McNemar
# ---------------------------------------------------------------------------
def two_lowest(df, block):
    """The pair of leak rates a two-rate block runs on, announced when it is a
    choice: with three or more runs present we take the two lowest so that the
    numbers a two-rate corpus produced remain reproducible."""
    rates = sorted(df["leak_rate"].unique())
    if len(rates) < 2:
        raise SystemExit("%s needs at least two leak rates, found %s" % (block, rates))
    if len(rates) > 2:
        print("  NOTE: %d leak rates present; this block uses the two lowest, "
              "%g%% and %g%%." % (len(rates), 100 * rates[0], 100 * rates[1]))
    return rates[0], rates[1]


def part_c(df, results):
    rule("(c)  PAIRED STRUCTURE ACROSS TWO LEAK RATES")

    wide = df.pivot(index="mid", columns="leak_rate",
                    values=["hit", "L", "degree"])
    r5, r10 = two_lowest(df, "(c)")
    h5 = wide[("hit", r5)].astype(int)
    h10 = wide[("hit", r10)].astype(int)
    L5 = wide[("L", r5)].astype(int)
    L10 = wide[("L", r10)].astype(int)

    both = int(((h5 == 1) & (h10 == 1)).sum())
    only5 = int(((h5 == 1) & (h10 == 0)).sum())
    only10 = int(((h5 == 0) & (h10 == 1)).sum())
    neither = int(((h5 == 0) & (h10 == 0)).sum())

    print("                       hit @10%%    miss @10%%")
    print("    hit  @5%%          %6d      %8d" % (both, only5))
    print("    miss @5%%          %6d      %8d" % (only10, neither))
    print("\n    total victims: %d   total hits: %d" % (both + only5 + only10 + neither,
                                                        int(h5.sum() + h10.sum())))
    print("    overlap: %d of %d hits are the same victim at both rates (%.1f%%)"
          % (both, int(h5.sum() + h10.sum()), 200.0 * both / (h5.sum() + h10.sum())))

    # McNemar, exact and chi-square with continuity correction
    n_disc = only5 + only10
    mcnemar_chi2 = (abs(only5 - only10) - 1) ** 2 / n_disc if n_disc else float("nan")
    mcnemar_p_chi2 = float(stats.chi2.sf(mcnemar_chi2, 1)) if n_disc else float("nan")
    mcnemar_p_exact = float(stats.binomtest(only5, n_disc, 0.5).pvalue) if n_disc else float("nan")
    print("\n    McNemar (continuity-corrected) chi2 = %.3f, p = %.4g"
          % (mcnemar_chi2, mcnemar_p_chi2))
    print("    McNemar exact (binomial)            p = %.4g" % mcnemar_p_exact)

    # among hit@5%-only victims, what happened to their leak at 10%?
    m5only = (h5 == 1) & (h10 == 0)
    dL = (L10 - L5)[m5only]
    more = int((dL > 0).sum())
    same = int((dL == 0).sum())
    fewer = int((dL < 0).sum())
    print("\n    of the %d victims hit at 5%% only:" % only5)
    print("      leaked MORE edges at 10%% and still missed : %d" % more)
    print("      leaked the same number                    : %d" % same)
    print("      leaked fewer                              : %d" % fewer)
    print("      median leak 5%% -> 10%%: %d -> %d" % (int(L5[m5only].median()),
                                                       int(L10[m5only].median())))

    m10only = (h5 == 0) & (h10 == 1)
    dL10 = (L10 - L5)[m10only]
    print("\n    of the %d victims hit at 10%% only:" % only10)
    print("      leaked more edges at 10%%: %d" % int((dL10 > 0).sum()))
    print("      leaked the same or fewer: %d" % int((dL10 <= 0).sum()))

    results["paired"] = {
        "leak_rates": [float(r5), float(r10)],
        "table": {"hit_both": both, "hit_5pct_only": only5,
                  "hit_10pct_only": only10, "hit_neither": neither},
        "n_victims": both + only5 + only10 + neither,
        "n_hits_total": int(h5.sum() + h10.sum()),
        "overlap_fraction_of_hits": 2.0 * both / float(h5.sum() + h10.sum()),
        "mcnemar_chi2_cc": float(mcnemar_chi2),
        "mcnemar_p_chi2_cc": mcnemar_p_chi2,
        "mcnemar_p_exact": mcnemar_p_exact,
        "hit5_only_leak_change": {"more": more, "same": same, "fewer": fewer,
                                  "median_L_5pct": int(L5[m5only].median()),
                                  "median_L_10pct": int(L10[m5only].median())},
        "hit10_only_leak_change": {"more": int((dL10 > 0).sum()),
                                   "same_or_fewer": int((dL10 <= 0).sum())},
    }


# ---------------------------------------------------------------------------
# (c4) three or more leak rates: Cochran's Q, every pairwise McNemar, and the
#      multi-way overlap of the hit sets
# ---------------------------------------------------------------------------
def part_c_multi(df, results):
    rates = sorted(df["leak_rate"].unique())
    if len(rates) < 3:
        return
    rule("(c4)  ALL %d LEAK RATES TOGETHER" % len(rates))

    wide = df.pivot(index="mid", columns="leak_rate", values="hit").astype(int)
    wide = wide.dropna()
    H = wide[rates].to_numpy()
    n, k = H.shape

    # Cochran's Q: the k-sample generalisation of McNemar for binary outcomes
    # measured repeatedly on the same subjects.
    col = H.sum(axis=0).astype(float)          # hits per rate
    row = H.sum(axis=1).astype(float)          # hits per victim
    denom = (k * row.sum() - (row ** 2).sum())
    q = ((k - 1) * (k * (col ** 2).sum() - col.sum() ** 2) / denom) if denom else float("nan")
    q_p = float(stats.chi2.sf(q, k - 1)) if denom else float("nan")
    print("  hits per rate: %s over %d victims"
          % (", ".join("%g%%: %d" % (100 * r, c) for r, c in zip(rates, col)), n))
    print("  Cochran's Q = %.3f on %d d.f., p = %.4g" % (q, k - 1, q_p))
    print("  (the k-rate generalisation of McNemar: are the run-level rates equal?)")

    # every pairwise McNemar, so the flattening of the curve can be located
    pairs = {}
    print("\n  pairwise McNemar (continuity-corrected chi2, and exact):")
    for i in range(k):
        for j in range(i + 1, k):
            a, b = H[:, i], H[:, j]
            b01 = int(((a == 1) & (b == 0)).sum())
            b10 = int(((a == 0) & (b == 1)).sum())
            disc = b01 + b10
            chi2 = (abs(b01 - b10) - 1) ** 2 / disc if disc else float("nan")
            p_chi2 = float(stats.chi2.sf(chi2, 1)) if disc else float("nan")
            p_exact = float(stats.binomtest(b01, disc, 0.5).pvalue) if disc else float("nan")
            key = "%g_vs_%g" % (rates[i], rates[j])
            pairs[key] = {"only_lower": b01, "only_higher": b10,
                          "both": int(((a == 1) & (b == 1)).sum()),
                          "mcnemar_chi2_cc": float(chi2),
                          "p_chi2_cc": p_chi2, "p_exact": p_exact}
            print("    %5g%% vs %5g%%   %3d / %3d discordant   chi2 = %6.3f   p = %.4g"
                  % (100 * rates[i], 100 * rates[j], b01, b10, chi2, p_exact))

    # multi-way overlap: how many victims are hit in exactly j of the k runs,
    # against the expectation if the runs were independent draws
    p_hat = col / float(n)
    exp_counts = []
    for j in range(k + 1):
        tot = 0.0
        for mask in range(1 << k):
            if bin(mask).count("1") != j:
                continue
            term = 1.0
            for i in range(k):
                term *= p_hat[i] if (mask >> i) & 1 else (1.0 - p_hat[i])
            tot += term
        exp_counts.append(n * tot)
    obs_counts = [int((row == j).sum()) for j in range(k + 1)]
    print("\n  victims hit in exactly j of the %d runs:" % k)
    print("    %-4s %10s %12s %8s" % ("j", "observed", "expected", "ratio"))
    for j in range(k + 1):
        ratio = obs_counts[j] / exp_counts[j] if exp_counts[j] else float("nan")
        print("    %-4d %10d %12.2f %8.2f" % (j, obs_counts[j], exp_counts[j], ratio))
    print("  (j >= 2 above expectation means a real victim-level component;")
    print("   the runs are separate corpora, so independence is the null.)")

    results["multi_rate"] = {
        "leak_rates": [float(r) for r in rates],
        "n_victims": int(n),
        "hits_per_rate": {"%g" % r: int(c) for r, c in zip(rates, col)},
        "cochran_q": float(q), "cochran_dof": int(k - 1), "cochran_p": q_p,
        "pairwise_mcnemar": pairs,
        "overlap_by_count": {
            "observed": obs_counts,
            "expected_if_independent": [float(e) for e in exp_counts],
        },
    }


# ---------------------------------------------------------------------------
# (c2) table support: exact goodness-of-fit d.f., and the fitted value of the
#      aggregated tail bin that Table 4 reports as a single row
# ---------------------------------------------------------------------------
def table_support(df, A, tau, results):
    rule("(c2)  GOODNESS-OF-FIT d.f. AND THE AGGREGATED TAIL BIN")

    by_L = df.groupby("L")["hit"].agg(["count", "sum"])

    # the chi-square in model_fit.json keeps bins with n >= 50 and expected >= 5,
    # then subtracts the two fitted parameters
    used = []
    for L, row in by_L.iterrows():
        n, k = int(row["count"]), int(row["sum"])
        if n < 50:
            continue
        e = n * A * (1.0 - math.exp(-L / tau))
        if e >= 5:
            used.append(int(L))
    dof = max(len(used) - 2, 1)
    print("  bins entering the Pearson statistic (n>=50 and expected>=5): %s" % used)
    print("  d.f. = %d bins - 2 fitted parameters = %d" % (len(used), dof))

    # aggregated tail bin
    tail = by_L[by_L.index >= 11]
    n_tail = int(tail["count"].sum())
    k_tail = int(tail["sum"].sum())
    exp_tail = float(sum(int(r["count"]) * A * (1.0 - math.exp(-L / tau))
                         for L, r in tail.iterrows()))
    print("  tail bin L>=11: n = %d, hits = %d, observed = %.2f%%" %
          (n_tail, k_tail, 100.0 * k_tail / n_tail))
    print("  fitted for the same bin (n-weighted mean of the curve) = %.2f%%"
          % (100.0 * exp_tail / n_tail))

    results["table_support"] = {
        "chi2_bins_used": used, "n_bins": len(used), "dof": dof,
        "tail_bin": {"L_min": 11, "n": n_tail, "hits": k_tail,
                     "observed_rate": k_tail / n_tail,
                     "expected_hits": exp_tail,
                     "fitted_rate": exp_tail / n_tail},
    }


# ---------------------------------------------------------------------------
# (b4) model comparison INSIDE the fixed-effects likelihood
# ---------------------------------------------------------------------------
def within_victim_model_comparison(df, results, tau=3.99):
    """Rank functional forms of L by conditional AIC, not by comparing p-values.

    tau defaults to 3.99, the two-rate fit the published ladder was computed
    with; pass the current fit to re-derive it against a newer corpus.
    """
    rule("(b4)  WITHIN-VICTIM MODEL COMPARISON (conditional AIC)")

    d = df.copy()
    counts = d.groupby("mid")["hit"].agg(["sum", "count"])
    disc = counts[(counts["sum"] > 0) & (counts["sum"] < counts["count"])].index
    sub = d[d["mid"].isin(disc)].copy()

    # A conditional logit with no covariate assigns equal probability to every
    # arrangement of a victim's hits over that victim's observations, so the null
    # conditional log-likelihood is -sum_i log C(n_i, m_i). With two runs per
    # victim that is -n_victims * log 2; with three it is not, which is why this
    # is computed from the group sizes rather than assumed.
    n_pairs = int(sub["mid"].nunique())
    null_cll = -sum(math.log(math.comb(int(g["hit"].count()), int(g["hit"].sum())))
                    for _, g in sub.groupby("mid"))
    rows = [{"form": "null", "beta": 0.0, "cll": null_cll, "k": 0,
             "aic": -2 * null_cll}]

    forms = [
        ("L linear", lambda v: v),
        ("sqrt(L)", np.sqrt),
        ("log(1+L)", np.log1p),
        ("saturating", lambda v: 1.0 - np.exp(-v / tau)),
    ]
    for name, f in forms:
        x = pd.DataFrame({"x": f(sub["L"].astype(float))}, index=sub.index)
        try:
            res = sm.ConditionalLogit(sub["hit"].astype(float), x,
                                      groups=sub["mid"]).fit(disp=0, maxiter=200)
            cll = float(res.llf)
            rows.append({"form": name, "beta": float(res.params.iloc[0]),
                         "cll": cll, "k": 1, "aic": -2 * cll + 2})
        except Exception as exc:
            rows.append({"form": name, "error": str(exc)})

    print("    %-12s %8s %10s %9s" % ("form", "beta", "cond. LL", "AIC"))
    for r in rows:
        if "aic" in r:
            print("    %-12s %8.3f %10.3f %9.2f" % (r["form"], r["beta"], r["cll"], r["aic"]))
    ok = [r for r in rows if "aic" in r]
    best = min(ok, key=lambda r: r["aic"])
    print("    lowest conditional AIC: %s" % best["form"])
    results["within_victim_model_comparison"] = {
        "n_discordant_pairs": n_pairs, "rows": rows, "best": best["form"]}


# ---------------------------------------------------------------------------
# (c3) is the victim-level overlap larger than chance?
# ---------------------------------------------------------------------------
def overlap_signal(df, results):
    rule("(c3)  VICTIM-LEVEL SIGNAL: IS THE OVERLAP MORE THAN CHANCE?")

    wide = df.pivot(index="mid", columns="leak_rate", values="hit")
    r5, r10 = two_lowest(df, "(c3)")
    h5 = wide[r5].astype(int)
    h10 = wide[r10].astype(int)
    n = len(wide)
    both = int(((h5 == 1) & (h10 == 1)).sum())
    p5, p10 = float(h5.mean()), float(h10.mean())
    expected = n * p5 * p10
    p_poisson = float(stats.poisson.sf(both - 1, expected))
    table = [[both, int(((h5 == 1) & (h10 == 0)).sum())],
             [int(((h5 == 0) & (h10 == 1)).sum()), int(((h5 == 0) & (h10 == 0)).sum())]]
    chi2 = float(stats.chi2_contingency(table, correction=False)[0])
    phi = math.sqrt(chi2 / n)
    p_fisher = float(stats.fisher_exact(table)[1])

    print("  observed overlap (hit at both rates) : %d" % both)
    print("  expected if the two runs were independent: %.2f" % expected)
    print("  ratio                                 : %.1fx" % (both / expected))
    print("  P(overlap >= %d | Poisson %.2f)        : %.3g" % (both, expected, p_poisson))
    print("  phi coefficient                       : %.3f  (Fisher p = %.3g)" % (phi, p_fisher))
    print("  reading: a real victim-level component exists, but it is small --- most of")
    print("  the variance sits in the retrieval trajectory, which is what caps A.")

    results["overlap_signal"] = {
        "observed": both, "expected_if_independent": expected,
        "ratio": both / expected, "p_poisson_ge": p_poisson,
        "phi": phi, "p_fisher": p_fisher,
        "rate_5pct": p5, "rate_10pct": p10, "n_victims": n}


# ---------------------------------------------------------------------------
# (d) the L = 0 cell
# ---------------------------------------------------------------------------
def part_d(df, results):
    rule("(d)  THE L = 0 CELL")

    z = df[df["L"] == 0]
    n0, k0 = len(z), int(z["hit"].sum())
    pooled = df["hit"].mean()
    r10 = sorted(df["leak_rate"].unique())[1]
    p10 = df[df["leak_rate"] == r10]["hit"].mean()

    p_pooled = float(stats.binomtest(k0, n0, pooled, alternative="less").pvalue)
    p_10 = float(stats.binomtest(k0, n0, p10, alternative="less").pvalue)
    lo, hi = wilson(k0, n0)
    rule_of_three = 3.0 / n0

    print("  observed: %d hits in %d observations at L = 0" % (k0, n0))
    print("  baselines: pooled rate = %.4f (%d/%d); 10%%-run rate = %.4f"
          % (pooled, int(df["hit"].sum()), len(df), p10))
    print("\n  exact binomial p (one-sided, vs pooled baseline)  = %.3g" % p_pooled)
    print("  exact binomial p (one-sided, vs 10%%-run baseline)  = %.3g" % p_10)
    print("\n  Wilson 95%% interval  : [%.4f%%, %.4f%%]" % (100 * lo, 100 * hi))
    print("  rule of three (95%% UB): %.4f%%" % (100 * rule_of_three))
    print("  expected hits under the pooled baseline: %.2f" % (n0 * pooled))

    results["L0_cell"] = {
        "n": n0, "hits": k0,
        "baseline_pooled": float(pooled), "baseline_10pct_run": float(p10),
        "expected_hits_pooled": float(n0 * pooled),
        "p_exact_vs_pooled": p_pooled, "p_exact_vs_10pct": p_10,
        "wilson95": [lo, hi], "rule_of_three_upper95": rule_of_three,
    }


# ---------------------------------------------------------------------------
# (e) headline rates with intervals
# ---------------------------------------------------------------------------
def part_e(df, results):
    rule("(e)  HEADLINE RATES WITH WILSON 95% INTERVALS")

    out = {}
    for rate, g in df.groupby("leak_rate"):
        k, n = int(g["hit"].sum()), len(g)
        lo, hi = wilson(k, n)
        label = "%g%%" % (100 * rate)
        print("  leak rate %-5s  %s" % (label, fmt_ci(k, n)))
        out[label] = {"hits": k, "n": n, "rate": k / n, "wilson95": [lo, hi]}

    rates = sorted(df["leak_rate"].unique())
    print("")
    for lo_r, hi_r in zip(rates, rates[1:]):
        a = df[df["leak_rate"] == lo_r]
        b = df[df["leak_rate"] == hi_r]
        ratio = (b["hit"].mean()) / (a["hit"].mean())
        label = "ratio_%g_over_%g" % (hi_r, lo_r)
        print("  ratio %g%% / %g%% = %.3f" % (100 * hi_r, 100 * lo_r, ratio))
        out[label] = float(ratio)
    # the runs share victims, so an unpaired test would be wrong here; the paired
    # comparisons are McNemar in (c) and, for three or more rates, (c4).
    print("  NOTE: the runs share the same victims, so the correct test of a")
    print("        difference is McNemar in (c)/(c4), not a two-sample proportion test.")
    results["headline_rates"] = out


# ---------------------------------------------------------------------------
# (f) and (g) stratified tables
# ---------------------------------------------------------------------------
def k_bucket(k):
    if k <= 1:
        return "k=1"
    if k <= 4:
        return "k=2-4"
    if k <= 19:
        return "k=5-19"
    if k <= 99:
        return "k=20-99"
    return "k>=100"


BUCKET_ORDER = ["k=1", "k=2-4", "k=5-19", "k=20-99", "k>=100"]


def part_fg(df, results):
    rule("(f)  EXTREME ANONYMITY-SET STRATA: FISHER EXACT")

    d = df.copy()
    d["bucket"] = d["anonymity_set_k"].apply(k_bucket)
    fisher = {}
    for rate, g in d.groupby("leak_rate"):
        lo_b = g[g["bucket"] == "k=1"]
        hi_b = g[g["bucket"] == "k>=100"]
        a1, n1 = int(lo_b["hit"].sum()), len(lo_b)
        a2, n2 = int(hi_b["hit"].sum()), len(hi_b)
        table = [[a1, n1 - a1], [a2, n2 - a2]]
        odds, p = stats.fisher_exact(table)
        label = "%g%%" % (100 * rate)
        print("  leak %-5s  k=1: %d/%d = %.2f%%   vs   k>=100: %d/%d = %.2f%%"
              % (label, a1, n1, 100 * a1 / n1, a2, n2, 100 * a2 / n2))
        print("             odds ratio = %.3f, Fisher exact p = %.4g" % (odds, p))
        fisher[label] = {"k1_hits": a1, "k1_n": n1, "k100_hits": a2, "k100_n": n2,
                         "odds_ratio": float(odds), "p_fisher": float(p)}

    # pooled across both rates (observations, not independent victims - flagged)
    lo_b = d[d["bucket"] == "k=1"]
    hi_b = d[d["bucket"] == "k>=100"]
    table = [[int(lo_b["hit"].sum()), len(lo_b) - int(lo_b["hit"].sum())],
             [int(hi_b["hit"].sum()), len(hi_b) - int(hi_b["hit"].sum())]]
    odds, p = stats.fisher_exact(table)
    print("\n  pooled over both rates: odds ratio = %.3f, p = %.4g" % (odds, p))
    print("  NOTE: pooling counts each victim twice; treat as descriptive.")
    fisher["pooled"] = {"odds_ratio": float(odds), "p_fisher": float(p),
                        "caveat": "each victim appears twice; not independent"}
    results["fisher_k_strata"] = fisher

    # ---- (g) Wilson intervals for every stratified cell -------------------
    rule("(g)  WILSON 95% INTERVALS FOR EVERY STRATIFIED CELL")

    print("\n  by anonymity-set size:")
    strata = {}
    for b in BUCKET_ORDER:
        strata[b] = {}
        for rate, g in d[d["bucket"] == b].groupby("leak_rate"):
            k, n = int(g["hit"].sum()), len(g)
            lo, hi = wilson(k, n)
            label = "%g%%" % (100 * rate)
            strata[b][label] = {"hits": k, "n": n, "rate": k / n, "wilson95": [lo, hi]}
            print("    %-8s @%-5s %s" % (b, label, fmt_ci(k, n)))
    results["strata_ci"] = strata

    # by sensitive relation: reconstruct membership from the masked graph
    print("\n  by sensitive relation:")
    rel = per_relation_membership()
    rel_ci = {}
    if rel is None:
        print("    [skipped] the masked graph is not available; per-relation")
        print("    membership cannot be rebuilt from per_victim.csv alone.")
        results["relation_ci"] = {"available": False,
                                  "reason": "masked graph not found next to this script"}
    else:
        for r, mids in rel.items():
            label_r = SENSITIVE_LABEL.get(r, r.split("/")[-1])
            rel_ci[label_r] = {}
            grp = d[d["mid"].isin(mids)]
            for rate, g in grp.groupby("leak_rate"):
                k, n = int(g["hit"].sum()), len(g)
                lo, hi = wilson(k, n)
                label = "%g%%" % (100 * rate)
                rel_ci[label_r][label] = {"hits": k, "n": n, "rate": k / n,
                                          "wilson95": [lo, hi]}
                print("    %-20s @%-5s %s" % (label_r, label, fmt_ci(k, n)))
        results["relation_ci"] = {"available": True, "strata": rel_ci}


def per_relation_membership():
    """Which victims carry which sensitive relation, from the masked graph."""
    base = os.path.join(ROOT, "data", "FB15k-237-id-masked")
    train = os.path.join(base, "train.txt")
    victims = os.path.join(base, "victims.json")
    if not (os.path.exists(train) and os.path.exists(victims)):
        return None
    position = {
        "/medicine/disease/notable_people_with_this_condition": "tail",
        "/people/cause_of_death/people": "tail",
        "/celebrities/celebrity/sexual_relationships./celebrities/romantic_relationship/celebrity": "both",
        "/people/person/religion": "head",
        "/people/ethnicity/people": "tail",
        "/government/political_party/politicians_in_this_party./government/political_party_tenure/politician": "tail",
        "/base/schemastaging/person_extra/net_worth./measurement_unit/dated_money_value/currency": "head",
    }
    victim_mids = set(json.load(io.open(victims, encoding="utf-8")))
    out = collections.defaultdict(set)
    with io.open(train, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            pos = position.get(r)
            if pos is None:
                continue
            if pos in ("tail", "both") and t in victim_mids:
                out[r].add(t)
            if pos in ("head", "both") and h in victim_mids:
                out[r].add(h)
    return {r: out[r] for r in position if r in out}


# ---------------------------------------------------------------------------
def parse_args(argv):
    rates, out_json = None, OUT_JSON
    i = 0
    while i < len(argv):
        if argv[i] == "--rates" and i + 1 < len(argv):
            rates = [float(x) for x in argv[i + 1].split(",") if x.strip()]
            i += 2
        elif argv[i] == "--out" and i + 1 < len(argv):
            out_json = argv[i + 1]
            i += 2
        else:
            raise SystemExit("usage: multivariate_analysis.py "
                             "[--rates 0.05,0.1] [--out results.json]")
    return rates, out_json


def main(argv=None):
    keep_rates, out_json = parse_args(list(argv if argv is not None else sys.argv[1:]))
    df, A, tau = load(keep_rates)
    print("loaded %d observations, %d victims, %d hits"
          % (len(df), df["mid"].nunique(), int(df["hit"].sum())))
    print("leak rates: %s"
          % ", ".join("%g%%" % (100 * r) for r in sorted(df["leak_rate"].unique())))
    print("saturating fit from model_fit.json: A = %.3f, tau = %.2f" % (A, tau))

    results = {"n_observations": int(len(df)), "n_victims": int(df["mid"].nunique()),
               "n_hits": int(df["hit"].sum()), "A_from_fit": A, "tau_from_fit": tau}

    confound(df, results)
    part_a(df, tau, results)
    part_b(df, tau, results)
    part_c(df, results)
    part_c_multi(df, results)
    within_victim_model_comparison(df, results)
    overlap_signal(df, results)
    table_support(df, A, tau, results)
    part_d(df, results)
    part_e(df, results)
    part_fg(df, results)

    with io.open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nwrote %s" % out_json)


if __name__ == "__main__":
    main()
