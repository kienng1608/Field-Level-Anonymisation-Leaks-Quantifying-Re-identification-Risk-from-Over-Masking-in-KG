"""
Data loading utilities.
Builds KG dict structure compatible with KG-GPT's evidence retrieval logic.
"""
import math
import os
from collections import defaultdict


def load_triples(filepath):
    """Load triples from tab-separated file."""
    triples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


def build_kg_dict(triples):
    """
    Build KG dictionary compatible with KG-GPT structure.
    
    KG[entity][relation] = [tails]
    KG[entity]["~"+relation] = [heads]   (reverse)
    
    This allows:
      KG["/m/EXAMPLE01"]["film"] → ["Serpico", "The Godfather"]
      KG["Serpico"]["~film"] → ["/m/EXAMPLE01", "Bryan Singer"]
    """
    KG = defaultdict(lambda: defaultdict(list))
    
    for h, r, t in triples:
        KG[h][r].append(t)
        KG[t]["~" + r].append(h)
    
    return KG


def load_wiki_mapping(filepath):
    """
    Load Freebase MID → human-readable name mapping.
    File format: freebase_id \t wikidata_id \t label
    """
    mid_to_name = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                mid_to_name[parts[0]] = parts[2]
    return mid_to_name


def load_all_train_triples(anon_data_dir, prune_geo=None):
    """
    Load train triples from anonymized dataset.

    When PRUNE_GEO_RELATIONS is on, place-to-place edges are dropped up front so
    they never reach retrieval — see config for the measured effect on
    GT-in-evidence. Pass prune_geo=False to load the raw graph (needed when
    measuring the dataset itself rather than running the attack).
    """
    from . import config as _cfg
    train_path = os.path.join(anon_data_dir, "train.txt")
    triples = load_triples(train_path)
    if prune_geo is None:
        prune_geo = getattr(_cfg, "PRUNE_GEO_RELATIONS", False)
    if prune_geo:
        geo = getattr(_cfg, "GEO_RELATION_PREFIXES", ())
        triples = [tr for tr in triples
                   if not any(g in tr[1] for g in geo)]
    return triples


def compute_kg_stats(triples, hub_percentile=None):
    """
    Compute one-time KG statistics used by Step 3 for evidence ranking
    and noise filtering.

    Args:
        hub_percentile: degree percentile above which an entity counts as a hub.
            Defaults to config.HUB_PERCENTILE so changing the config actually
            takes effect (it used to be hard-coded to 95.0 here, which silently
            ignored the config value at every call site).

    Returns dict with keys:
        - degree: dict entity -> total degree (in + out)
        - hub_threshold: float; entity is "hub" if total_connections > threshold
        - relation_idf: dict relation -> idf score (log(N / freq[rel]))
        - person_entities: set of entities that participate in at least one
          relation under /people/... namespace (used to prune multi-hop)
    """
    if hub_percentile is None:
        from .config import HUB_PERCENTILE
        hub_percentile = HUB_PERCENTILE
    degree = defaultdict(int)
    rel_freq = defaultdict(int)
    person_entities = set()

    for h, r, t in triples:
        degree[h] += 1
        degree[t] += 1
        rel_freq[r] += 1
        if r.startswith("/people/"):
            person_entities.add(h)
            person_entities.add(t)

    n_triples = len(triples)
    relation_idf = {}
    for r, f in rel_freq.items():
        # Smoothed IDF: rare relations score higher.
        relation_idf[r] = math.log((n_triples + 1.0) / (f + 1.0))

    if degree:
        try:
            import numpy as np
            hub_threshold = float(np.percentile(list(degree.values()), hub_percentile))
        except ImportError:
            sorted_deg = sorted(degree.values())
            idx = int(len(sorted_deg) * hub_percentile / 100.0)
            idx = min(idx, len(sorted_deg) - 1)
            hub_threshold = float(sorted_deg[idx])
    else:
        hub_threshold = 500.0

    # Floor the hub threshold so very small graphs still filter the obvious hubs.
    hub_threshold = max(hub_threshold, 50.0)

    return {
        "degree": dict(degree),
        "hub_threshold": hub_threshold,
        "relation_idf": relation_idf,
        "person_entities": person_entities,
    }
