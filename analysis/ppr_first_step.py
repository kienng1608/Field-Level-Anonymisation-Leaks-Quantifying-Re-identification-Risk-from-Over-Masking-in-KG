"""Why Personalized PageRank misses: where its first-step mass goes.

Established: P2(m_v->c) ranks exactly like RA (300/300), so the code is faithful.
Remaining question: where does PPR's mass go instead?

Measures, per victim at several alphas:
  - share of PPR mass on identified keys that are DIRECT neighbours of m_v
  - top-1 rate, and top-1 rate after excluding direct person neighbours
"""
import importlib.util, os, sys, collections
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
spec = importlib.util.spec_from_file_location('rb', 'baselines/run_non_llm_baselines.py')
rb = importlib.util.module_from_spec(spec)
sys.modules['rb'] = rb
spec.loader.exec_module(rb)

triples, victims = rb.load_dataset('15')
G = rb.build_graph_structures(triples)
P_T = G['P_T']; n = P_T.shape[0]
n2i = G['node_to_id']
ind_idx = np.asarray(G['ind_indices'])
ind_names = np.asarray(G['all_ind_nodes'])
nbrs = G['neighbors']
pos = {nm: i for i, nm in enumerate(ind_names)}

mids = [m for m in victims if m in n2i][:500]


def ppr(ti, alpha, iters=120):
    p = np.zeros(n)
    p[ti] = 1.0
    for _ in range(iters):
        p = alpha * (P_T.dot(p))
        p[ti] += (1.0 - alpha)
    return p


print('%-7s %-14s %-16s %-16s' % ('alpha', 'top-1', 'top-1 excl.direct', 'mass on direct'))
for alpha in (0.85, 0.5, 0.2):
    hit = hitx = tot = 0
    frac = []
    for m in mids:
        true = victims[m]
        sc = ppr(n2i[m], alpha)[ind_idx]
        tot += 1
        bi = int(np.argmax(sc))
        if ind_names[bi] == true and np.sum(sc == sc[bi]) == 1:
            hit += 1
        direct = {x for x in nbrs[m] if x.startswith('IND_')}
        tot_mass = sc.sum()
        if tot_mass > 0:
            dm = sum(sc[pos[d]] for d in direct if d in pos)
            frac.append(dm / tot_mass)
        if direct:
            mask = np.array([nm not in direct for nm in ind_names])
            sc2 = np.where(mask, sc, -1.0)
            b2 = int(np.argmax(sc2))
            if ind_names[b2] == true and np.sum(sc2 == sc2[b2]) == 1:
                hitx += 1
        else:
            if ind_names[bi] == true and np.sum(sc == sc[bi]) == 1:
                hitx += 1
    print('%-7.2f %-14s %-16s %-16s'
          % (alpha, '%.2f%%' % (100*hit/tot), '%.2f%%' % (100*hitx/tot),
             '%.1f%%' % (100*np.mean(frac))))

print()
print('victims having at least one identified key as a DIRECT neighbour of m_v:')
d = sum(1 for m in mids if any(x.startswith('IND_') for x in nbrs[m]))
print('   %d/%d = %.1f%%' % (d, len(mids), 100*d/len(mids)))
