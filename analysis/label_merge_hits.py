"""How many RA / AA hits rest on a node created by the label merge (entities
sharing a Wikidata label collapsed into one node)? A hit 'rests solely' on
merged nodes if every anchor m_v and c_v share is merged; it 'involves' one if
any shared anchor is merged. Aggregate counts only."""
import collections
import glob
import io
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

raw = []
for f in sorted(glob.glob(os.path.join(DATA, "FB15k-237", "*.txt"))):
    if f.endswith("README.txt"):
        continue
    for line in io.open(f, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) == 3:
            raw.append(p)
ents = {e for h, r, t in raw for e in (h, t)}
lab = {}
for line in io.open(os.path.join(DATA, "FB15k-237", "fb_wiki_mapping.tsv"),
                    encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 3 and p[0] in ents:
        lab[p[0]] = p[2]
byl = collections.defaultdict(set)
for m, l in lab.items():
    byl[l].add(m)
merged = {l for l, v in byl.items() if len(v) > 1}

pub = json.load(io.open(os.path.join(BASE, "deanon_results",
                                     "non_llm_baselines_per_victim.json"),
                        encoding="utf-8"))
for rate in ("05", "10", "15"):
    d = os.path.join(DATA, "FB15k-237-id-move%s" % rate)
    vic = json.load(io.open(os.path.join(d, "victims.json"), encoding="utf-8"))
    nb = collections.defaultdict(set)
    for line in io.open(os.path.join(d, "train.txt"), encoding="utf-8"):
        h, r, t = line.rstrip("\n").split("\t")
        nb[h].add(t)
        nb[t].add(h)
    present = merged & set(nb)
    out = []
    for m in ("resource_allocation", "adamic_adar"):
        hits = [s for s, v in pub["rate_" + rate].items() if v[m]]
        solely = involve = 0
        for s in hits:
            sh = nb[s] & nb[vic[s]]
            if sh and all(a in present for a in sh):
                solely += 1
            if any(a in present for a in sh):
                involve += 1
        out.append("%s: %d hits, rest solely on a merged node %d, involve one %d"
                   % ("RA" if m.startswith("res") else "AA", len(hits), solely,
                      involve))
    print("x=%s%% (%d merged nodes in corpus): %s" % (rate, len(present),
                                                     " | ".join(out)))
