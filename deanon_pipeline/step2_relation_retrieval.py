"""
Step 2: Relation Retrieval
Find candidate relations for target MID, then use LLM to select top-K.
(Adapted from KG-GPT Step 2: Relation Retrieval)
"""
import os
import re
import time
from collections import defaultdict
from openai import OpenAI

from .config import (
    SENSITIVE_RELATIONS, BASE_URL, MODEL, NVIDIA_API_KEY,
    TOP_K_RELATIONS, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_SEED_KWARGS,
    PROMPTS_DIR, safe_max_tokens, LLM_REQUEST_TIMEOUT,
)


def get_candidate_relations(KG, target_mid, sensitive_contexts, kg_stats=None,
                            min_selectivity=0.0, max_candidates=None):
    """
    Collect candidate relations connected to target_mid and its sensitive
    context entities.

    Pruning, in order of preference:
      1. (data-driven) selectivity score = #distinct_tails / #triples_with_rel.
         Relations with low selectivity (everyone shares the same value, e.g.
         /location/country/currency_used → "United States dollar") are uninformative
         for identification. When `kg_stats` is supplied, drop relations whose
         selectivity falls below `min_selectivity`.
      2. (hard-coded fallback) if no kg_stats, drop a small list of well-known
         statistical/financial namespaces.

    Args:
        KG: adjacency dict
        target_mid: target's MID
        sensitive_contexts: list of dicts (relation + context_entity)
        kg_stats: optional dict from data_loader.compute_kg_stats() — must
            contain "relation_idf" (we recompute selectivity from KG).
            When None, uses keyword fallback.
        min_selectivity: drop relations whose selectivity < this threshold
            (default 0.0 keeps everything that passes the keyword fallback).
        max_candidates: cap the candidate set after pruning (sorted by
            selectivity desc when kg_stats is supplied).

    Returns:
        list[str] of relation names.
    """
    NOISY_KEYWORDS = [
        "currency", "tuition", "gdp", "gni", "assets", "operating_income",
        "revenue", "net_worth", "budget", "box_office",
    ]

    def keyword_filter(rel):
        rel_lower = rel.lower()
        return not any(kw in rel_lower for kw in NOISY_KEYWORDS)

    raw = set()
    if target_mid in KG:
        raw.update(KG[target_mid].keys())
    for ctx in sensitive_contexts:
        ctx_ent = ctx["context_entity"]
        if ctx_ent in KG:
            raw.update(KG[ctx_ent].keys())

    if kg_stats is None:
        # Legacy keyword-only path
        filtered = [r for r in raw if keyword_filter(r)]
        if max_candidates is not None:
            filtered = filtered[:max_candidates]
        return filtered

    # Data-driven selectivity prefilter
    # selectivity = #distinct tails / #triples per relation, computed from KG.
    # We compute this once on demand and cache via attr on the kg_stats dict.
    if "relation_selectivity" not in kg_stats:
        rel_tail_counts = defaultdict(int)
        rel_distinct_tails = defaultdict(set)
        for ent in KG:
            for rel, tails in KG[ent].items():
                actual = rel.lstrip("~")
                rel_tail_counts[actual] += len(tails)
                rel_distinct_tails[actual].update(tails)
        selectivity = {}
        for rel, n_triples in rel_tail_counts.items():
            distinct = len(rel_distinct_tails[rel])
            selectivity[rel] = distinct / n_triples if n_triples else 0.0
        kg_stats["relation_selectivity"] = selectivity

    selectivity = kg_stats["relation_selectivity"]

    scored = []
    for rel in raw:
        if not keyword_filter(rel):
            continue
        actual = rel.lstrip("~")
        s = selectivity.get(actual, 0.0)
        if s < min_selectivity:
            continue
        scored.append((s, rel))

    # Sort by selectivity desc (most informative first), stable on rel
    scored.sort(key=lambda x: (-x[0], x[1]))
    out = [rel for _, rel in scored]
    if max_candidates is not None:
        out = out[:max_candidates]
    return out


def clean_relation_for_display(rel):
    """Keep Freebase relation path exactly as-is."""
    return rel


def _format_context(sensitive_contexts):
    """
    Render sensitive_contexts as a human-readable single line that the LLM
    can use to reason about who the anonymized person likely is.
    """
    if not sensitive_contexts:
        return "(no public sensitive context)"
    parts = []
    for ctx in sensitive_contexts:
        rel = ctx["relation"]
        ent = ctx["context_entity"]
        pos = ctx.get("position", "tail")
        # Render direction: "rel -> ent" if target is head, "rel <- ent" if target is tail
        arrow = "->" if pos == "head" else "<-"
        parts.append(f"{rel} {arrow} {ent}")
    return "; ".join(parts)


def build_relation_retrieval_prompt(target_mid, sensitive_contexts,
                                    candidate_relations, top_k):
    """
    Build prompt for LLM relation selection.

    Replaces all 3 placeholders in the prompt template:
      <<<TARGET_MID>>>, <<<CONTEXT>>>, <<<RELATION_SET>>>, <<<TOP_K>>>
    """
    prompt_path = os.path.join(PROMPTS_DIR, "relation_retrieval_prompt.txt")
    template = open(prompt_path, 'r', encoding='utf-8').read()

    rel_display = [clean_relation_for_display(r) for r in candidate_relations]
    context_str = _format_context(sensitive_contexts)

    prompt = template.replace("<<<TOP_K>>>", str(top_k))
    prompt = prompt.replace("<<<TARGET_MID>>>", target_mid)
    prompt = prompt.replace("<<<CONTEXT>>>", context_str)
    prompt = prompt.replace("<<<RELATION_SET>>>", str(rel_display))

    return prompt


def parse_relation_selection(answer, candidate_relations):
    """
    Parse LLM response to extract selected relations.
    Reuses KG-GPT's retrieval_relation_parse_answer() logic.
    
    Returns the ORIGINAL relation strings (not display names).
    """
    # Extract list from brackets: ['rel1', 'rel2']
    pattern = r'\[[^\]]+\]'
    matches = re.findall(pattern, answer)
    
    if len(matches) == 0:
        return candidate_relations[:TOP_K_RELATIONS]  # Fallback
    
    # Parse components
    components = []
    for match in matches:
        items = match.strip('[]').split(',')
        for item in items:
            item = item.strip().strip("'\"")
            if item:
                components.append(item)
    
    # Map display names back to original relation strings
    selected = []
    for comp in components:
        comp_lower = comp.lower().strip()
        for rel in candidate_relations:
            display = clean_relation_for_display(rel).lower()
            if comp_lower == display or comp_lower in display:
                if rel not in selected:
                    selected.append(rel)
                    break
    
    # If mapping failed, return first top_k candidates
    if not selected:
        return candidate_relations[:TOP_K_RELATIONS]
    
    return selected


def retrieve_relations(KG, target_mid, sensitive_contexts, top_k=None, api_key=None,
                       kg_stats=None, max_candidates_for_llm=50,
                       min_selectivity=0.01):
    """
    Full Step 2: Find candidates → LLM selects top-K.

    Args:
        kg_stats: optional dict from compute_kg_stats() — enables data-driven
            selectivity prefilter (replaces keyword blacklist).
        max_candidates_for_llm: cap how many candidates we put in front of the
            LLM. Pre-ranked by selectivity descending so the LLM sees the most
            informative relations first.
        min_selectivity: drop relations whose selectivity < threshold (only
            applies when kg_stats is provided).

    Returns:
        selected_relations: list of relation strings
        candidate_relations: list of candidates actually presented to LLM
    """
    if top_k is None:
        top_k = TOP_K_RELATIONS
    if api_key is None:
        api_key = NVIDIA_API_KEY

    # 2a. Get candidate relations (with optional selectivity prefilter)
    candidate_relations = get_candidate_relations(
        KG, target_mid, sensitive_contexts, kg_stats=kg_stats,
        min_selectivity=min_selectivity,
        max_candidates=max_candidates_for_llm,
    )

    if not candidate_relations:
        return [], []

    if len(candidate_relations) <= top_k:
        return candidate_relations, candidate_relations
    
    # 2b. LLM selection
    prompt = build_relation_retrieval_prompt(
        target_mid, sensitive_contexts, candidate_relations, top_k
    )
    
    client = OpenAI(base_url=BASE_URL, api_key=api_key, timeout=LLM_REQUEST_TIMEOUT)

    call_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    selected = None
    for attempt in range(3):
        try:
            _t0 = time.time()
            response = client.chat.completions.create(
                model=MODEL,
                messages=call_messages,
                max_tokens=safe_max_tokens(call_messages),
                temperature=LLM_TEMPERATURE,
                top_p=0.1,
                **LLM_SEED_KWARGS,
            )
            print(f"  [TIMING] step2 (target relations) call: {time.time()-_t0:.1f}s")
            answer = response.choices[0].message.content
            selected = parse_relation_selection(answer, candidate_relations)
            if selected:
                break
        except Exception as e:
            print(f"  [LLM ERROR attempt {attempt+1}] {e}")
            time.sleep(2)
    
    if not selected:
        selected = candidate_relations[:top_k]

    return selected, candidate_relations


def retrieve_relations_for_placeholder(KG, placeholder_mid, target_facts,
                                        kg_stats=None, top_k=5, api_key=None,
                                        max_candidates_for_llm=30,
                                        min_selectivity=0.0):
    """
    Step 2 anchored on a PLACEHOLDER MID (not the TARGET).

    Used by hierarchical entity resolution: when the LLM requests EXPLORE
    or IDENTIFY on a placeholder, we run a focused Step 2 to pick the most
    discriminative relations for that placeholder, then expand evidence
    through them. This is fair (no wiki_mapping) — we only use the
    placeholder MID's adjacency in the anonymized KG.

    Args:
        KG: adjacency dict
        placeholder_mid: MID of the placeholder to characterize
        target_facts: list of strings describing TARGET (used in prompt to
            give LLM context about which relations matter)
        kg_stats: optional dict with relation_selectivity (reused from Step 2)
        top_k: how many relations to select for this placeholder
        api_key: NVIDIA API key
        max_candidates_for_llm: cap on candidates shown to LLM
        min_selectivity: drop relations below this selectivity

    Returns:
        (selected_relations, candidate_relations) — same shape as retrieve_relations
    """
    if top_k is None:
        top_k = TOP_K_RELATIONS
    if api_key is None:
        api_key = NVIDIA_API_KEY

    # Build fake sensitive_contexts using target_facts so prompt has context.
    # We pass empty list — placeholder's own KG keys provide the candidates.
    candidate_relations = get_candidate_relations(
        KG, placeholder_mid, sensitive_contexts=[], kg_stats=kg_stats,
        min_selectivity=min_selectivity,
        max_candidates=max_candidates_for_llm,
    )

    if not candidate_relations:
        return [], []

    if len(candidate_relations) <= top_k:
        return candidate_relations, candidate_relations

    # Build a placeholder-specific prompt: tell LLM that this MID is a
    # neighbor of the target, and we want relations that best characterize THIS placeholder.
    target_context_str = "; ".join(target_facts) if target_facts else "(no target context)"
    rel_display = [clean_relation_for_display(r) for r in candidate_relations]
    prompt = (
        f"You are helping de-anonymize a Knowledge Graph.\n"
        f"The TARGET person is connected via:\n  {target_context_str}\n\n"
        f"There is an ANONYMIZED PLACEHOLDER {placeholder_mid} who is a neighbor of TARGET.\n"
        f"From the relations connected to this placeholder, select the {top_k} most useful "
        f"for identifying WHO this placeholder is (their unique facts, distinctive ties).\n"
        f"Return ONLY a Python list like: ['rel1', 'rel2', ...]\n\n"
        f"Candidate relations:\n{rel_display}\n"
    )

    client = OpenAI(base_url=BASE_URL, api_key=api_key, timeout=LLM_REQUEST_TIMEOUT)
    call_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "/no_think\n" + prompt},
    ]
    selected = None
    for attempt in range(2):  # fewer retries — sub-task, fail-soft to top-K by selectivity
        try:
            _t0 = time.time()
            response = client.chat.completions.create(
                model=MODEL,
                messages=call_messages,
                max_tokens=safe_max_tokens(call_messages, floor=128, ceiling=512),
                temperature=LLM_TEMPERATURE,
                top_p=0.1,
                **LLM_SEED_KWARGS,
            )
            print(f"  [TIMING] step2 (placeholder relations) call: {time.time()-_t0:.1f}s")
            answer = response.choices[0].message.content or ""
            selected = parse_relation_selection(answer, candidate_relations)
            if selected:
                break
        except Exception as e:
            print(f"  [PH-LLM error attempt {attempt+1}] {e}")
            time.sleep(2)

    if not selected:
        selected = candidate_relations[:top_k]

    return selected[:top_k], candidate_relations
