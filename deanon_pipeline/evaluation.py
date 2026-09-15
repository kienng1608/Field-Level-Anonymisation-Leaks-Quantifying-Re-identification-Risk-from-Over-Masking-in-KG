"""
Evaluation module.
Compare LLM predictions against ground truth (wiki_mapping).
"""
import re
import unicodedata
from collections import defaultdict


def normalize_name(name):
    """
    Normalize name for comparison.
    Lowercase, strip diacritics, collapse whitespace, drop punctuation.
    """
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def check_match(predictions, real_name, aliases=None, strict=True):
    """
    Check if real_name (or any alias) appears in predictions.

    Args:
        predictions: list of predicted names
        real_name: ground-truth canonical name
        aliases: optional list of acceptable alternative names
        strict: when True (default), require exact normalized match.
                When False, fall back to legacy substring/parts heuristics
                (used only for baseline comparison).

    Returns:
        dict: {"match": bool, "rank": int (-1 if no match)}
    """
    if not real_name or not predictions:
        return {"match": False, "rank": -1}

    candidates_norm = {normalize_name(real_name)}
    if aliases:
        for a in aliases:
            n = normalize_name(a)
            if n:
                candidates_norm.add(n)
    candidates_norm.discard("")

    for rank, pred in enumerate(predictions, 1):
        pred_norm = normalize_name(pred)
        if not pred_norm:
            continue

        if pred_norm in candidates_norm:
            return {"match": True, "rank": rank}

        if strict:
            continue

        for cand in candidates_norm:
            if cand in pred_norm or pred_norm in cand:
                return {"match": True, "rank": rank}
            cand_parts = cand.split()
            if len(cand_parts) >= 2 and all(p in pred_norm for p in cand_parts):
                return {"match": True, "rank": rank}

    return {"match": False, "rank": -1}


def compute_metrics(results):
    """
    Compute evaluation metrics over all attack results.

    Args:
        results: list of dicts with keys "match", "rank", "mid", "real_name",
                 "sensitive_contexts"; optional key "structurally_indistinguishable"
                 (bool).

    Returns:
        dict of metrics
    """
    n_total = len(results)
    if n_total == 0:
        return {}

    n_top1 = sum(1 for r in results if r["match"] and r["rank"] == 1)
    n_topk = sum(1 for r in results if r["match"])
    mrr_sum = sum(1.0 / r["rank"] for r in results if r["match"] and r["rank"] > 0)

    # Per-relation breakdown
    per_relation = defaultdict(lambda: {"total": 0, "hit": 0})
    for r in results:
        for ctx in r.get("sensitive_contexts", []):
            rel = ctx["relation"]
            per_relation[rel]["total"] += 1
            if r["match"]:
                per_relation[rel]["hit"] += 1

    metrics = {
        "total_targets": n_total,
        "top1_hits": n_top1,
        "topk_hits": n_topk,
        "top1_accuracy": n_top1 / n_total,
        "topk_accuracy": n_topk / n_total,
        "mrr": mrr_sum / n_total,
        "per_relation": dict(per_relation),
    }

    n_indistinguishable = sum(
        1 for r in results if r.get("structurally_indistinguishable")
    )
    if n_indistinguishable:
        metrics["structurally_indistinguishable"] = n_indistinguishable

    return metrics


def print_metrics(metrics):
    """Pretty-print evaluation metrics."""
    print(f"\n{'━'*60}")
    print(f"  📋 ATTACK EVALUATION RESULTS")
    print(f"{'━'*60}")
    
    if not metrics:
        print("  Không đánh giá được (0 targets attacked).")
        print(f"{'━'*60}")
        return
        
    print(f"  Targets attacked:    {metrics['total_targets']}")
    print(f"  Top-1 Hits:          {metrics['top1_hits']}")
    print(f"  Top-K Hits:          {metrics['topk_hits']}")
    print(f"  Top-1 Accuracy:      {metrics['top1_accuracy']:.1%}")
    print(f"  Top-K Accuracy:      {metrics['topk_accuracy']:.1%}")
    print(f"  MRR:                 {metrics['mrr']:.4f}")
    if "structurally_indistinguishable" in metrics:
        print(f"  Indistinguishable:   {metrics['structurally_indistinguishable']} "
              f"(structurally identical to other candidates)")
    
    if metrics.get("per_relation"):
        print(f"\n  Per-Relation Breakdown:")
        for rel, counts in metrics["per_relation"].items():
            rel_short = rel.split("/")[-1]
            acc = counts["hit"] / counts["total"] if counts["total"] > 0 else 0
            print(f"    {rel_short:<30} {counts['hit']}/{counts['total']} ({acc:.0%})")
    print(f"{'━'*60}")
