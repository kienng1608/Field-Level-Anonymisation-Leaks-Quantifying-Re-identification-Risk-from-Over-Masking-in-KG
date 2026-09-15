"""
Verify that each predicted name is actually SUPPORTED BY THE GRAPH.

The closed-book prompt asks the model to cite evidence lines ("GROUNDING: facts
3, 6, 8"). Nothing enforces it, and a model that recognises a celebrity can
produce a fluent citation that does not hold up. Measured example (gemini,
/m/EXAMPLE01): it cited facts 1, 2, 3, 146 for a named musician, but fact 146 only
says that person plays saxophone — the same as 197 other people in that evidence —
while the polio link came from its own memory ("contracted polio in childhood",
a phrase that appears nowhere in the graph).

This script re-checks every HIT against the evidence the model actually saw:

  1. IN_EVIDENCE   — does the predicted name appear in the evidence at all?
  2. CITED         — do the cited line numbers exist (not hallucinated)?
  3. NAME_IN_CITED — does at least one cited line actually contain the name?
  4. CONNECTED     — do the cited lines form a path from the name to the target
                     in the evidence subgraph?

A prediction that fails any of these was not derived from the graph, whatever
its rationale claims.

Usage:
  python verify_grounding.py deanon_results/exp210_cb_move05.json
  python verify_grounding.py deanon_results/exp210_cb_move05.json --show 5
"""
import argparse
import collections
import io
import json
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TARGET_TOKEN = "[TARGET PERSON]"


def parse_evidence_lines(user_messages):
    """Map evidence line number -> the triple text on it.

    Only lines from the real evidence block count. The prompt also contains
    worked EXAMPLES with their own numbered triples (Example C opens with
    "1. ['[TARGET PERSON]', ...religion', 'Pentecostalism']"), and those would
    otherwise overwrite the target's real facts 1-4 and make citations look
    valid — or invalid — for the wrong reason. So parsing starts at the
    "EVIDENCE COLLECTED SO FAR" header and stops at the "=== EXAMPLES ===" one.

    Rounds re-print the whole block and append [NEW] lines, so a number can
    appear more than once; keep the LAST occurrence, which is the numbering in
    force when the model answered.
    """
    lines = {}
    for msg in user_messages:
        in_evidence = "EVIDENCE COLLECTED SO FAR" not in msg  # deltas have no header
        for raw in msg.split("\n"):
            if "EVIDENCE COLLECTED SO FAR" in raw:
                in_evidence = True
                continue
            if raw.lstrip().startswith("=== EXAMPLES"):
                in_evidence = False
                continue
            if not in_evidence:
                continue
            m = re.match(r"\s*(?:\[NEW\]\s*)?(\d+)\.\s*(\[.*\])\s*$", raw)
            if m:
                lines[int(m.group(1))] = m.group(2)
    return lines


def triple_from_line(text):
    """['head', 'relation', 'tail'] -> (head, relation, tail)."""
    parts = re.findall(r"'((?:[^'\\]|\\.)*)'", text)
    if len(parts) >= 3:
        return parts[0], parts[1], parts[-1]
    return None


def last_prediction_block(assistant_messages, name):
    """The rationale text of the round where `name` was ranked #1."""
    for msg in reversed(assistant_messages):
        if re.search(r"PREDICTIONS:\s*1\.\s*" + re.escape(name), msg):
            m = re.search(
                r"1\.\s*" + re.escape(name) + r".*?(?=\n\s*2\.|\nRETRIEVAL_REQUEST|$)",
                msg, re.S)
            return m.group(0) if m else msg
    return None


def cited_numbers(block):
    """Line numbers the rationale rests on.

    The prompt asks for a trailing "GROUNDING: facts 1, 3, 47", but the model also
    cites inline — "TARGET won an award shared with Michael Balcon (fact 3)" — and
    sometimes only that way. Counting just the trailing form would score a fully
    cited answer as ungrounded, so both are collected.
    """
    out = set()
    if not block:
        return out
    for m in re.finditer(r"GROUNDING:\s*facts?\s*([0-9,\s and]+)", block, re.I):
        out |= {int(z) for z in re.findall(r"\d+", m.group(1))}
    for m in re.finditer(r"\bfacts?\s+([\d,\s]*\d)", block, re.I):
        out |= {int(z) for z in re.findall(r"\d+", m.group(1))}
    return out


def connects(cited_triples, name, evidence_lines):
    """True if the cited lines link `name` to the target.

    Walks the little subgraph made of just the cited triples. The target is any
    node containing the [TARGET PERSON] token; the answer must be reachable from
    it through those lines alone.
    """
    adj = collections.defaultdict(set)
    for h, _, t in cited_triples:
        adj[h].add(t)
        adj[t].add(h)
    starts = [n for n in adj if TARGET_TOKEN in n]
    if not starts or name not in adj:
        return False
    seen, stack = set(starts), list(starts)
    while stack:
        cur = stack.pop()
        if cur == name:
            return True
        for nxt in adj[cur]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def competing_names(cited_triples, name, evidence_lines):
    """Other named people who satisfy the citation as fully as `name` does.

    An identification works by INTERSECTION: each bridge on its own is allowed to
    be broad (a disease holds ~30 people, an instrument ~200), so counting one
    bridge's members says nothing. What matters is how many people sit on ALL of
    the bridges the target itself touches — that is the set the citation actually
    narrows to.

    Only bridges the TARGET is cited as connected to are constraints; a bridge
    reached solely via the candidate (e.g. "Sanborn -> saxophone" when the target
    was never linked to saxophone) is the candidate's own attribute, not a shared
    one, so it cannot narrow anything.
    """
    target_bridges = {n for tr in cited_triples for n in (tr[0], tr[2])
                      if TARGET_TOKEN not in n and n != name
                      and any(TARGET_TOKEN in x for x in (tr[0], tr[2]))}
    if not target_bridges:
        return 0

    on_bridge = collections.defaultdict(set)   # bridge -> named people on it
    for text in evidence_lines.values():
        tr = triple_from_line(text)
        if not tr:
            continue
        h, _, t = tr
        for person, bridge in ((h, t), (t, h)):
            if bridge in target_bridges and TARGET_TOKEN not in person \
                    and not person.startswith("/m/") \
                    and not person.startswith("[PERSON_"):
                on_bridge[bridge].add(person)

    if not on_bridge:
        return 0
    survivors = set.intersection(*on_bridge.values()) if len(on_bridge) > 1 \
        else next(iter(on_bridge.values()))
    return len(survivors - {name})


def verify_one(result):
    gt = result.get("real_name", "")
    conv = result.get("conversation") or []
    users = [m["content"] for m in conv if m.get("role") == "user"]
    bots = [m["content"] for m in conv if m.get("role") == "assistant"]

    evidence_lines = parse_evidence_lines(users)
    max_line = max(evidence_lines) if evidence_lines else 0
    all_evidence = "\n".join(users)

    block = last_prediction_block(bots, gt)
    cited = cited_numbers(block)
    real_cited = {n for n in cited if n in evidence_lines}
    fake_cited = cited - real_cited
    cited_triples = [tr for tr in (triple_from_line(evidence_lines[n]) for n in real_cited) if tr]

    # How many OTHER named people reach the target through the same cited
    # bridges? A chain that also admits 197 saxophonists explains nothing about
    # this particular person, so count the rivals the citation leaves standing.
    rivals = competing_names(cited_triples, gt, evidence_lines)

    return {
        "mid": result.get("mid"),
        "name": gt,
        "in_evidence": gt in all_evidence,
        "n_cited": len(cited),
        "fake_cited": sorted(fake_cited),
        "name_in_cited": any(gt in line for line in
                             (evidence_lines[n] for n in real_cited)),
        "connected": connects(cited_triples, gt, evidence_lines),
        "rivals": rivals,
        "max_line": max_line,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_file")
    ap.add_argument("--show", type=int, default=0,
                    help="print this many failing cases in detail")
    ap.add_argument("--max-rivals", type=int, default=10,
                    help="how many other named people a citation's bridge nodes "
                         "may also admit before it stops being discriminative "
                         "(default 10)")
    args = ap.parse_args()

    data = json.load(io.open(args.results_file, encoding="utf-8"))
    results = data.get("results", data) if isinstance(data, dict) else data
    hits = [r for r in results if r.get("match")]

    checks = [verify_one(r) for r in hits]
    n = len(checks)
    if not n:
        print("No hits to verify.")
        return

    in_ev = [c for c in checks if c["in_evidence"]]
    no_fake = [c for c in in_ev if not c["fake_cited"]]
    named = [c for c in no_fake if c["name_in_cited"]]
    conn = [c for c in named if c["connected"]]
    uniq = [c for c in conn if c["rivals"] <= args.max_rivals]

    print(f"\n{os.path.basename(args.results_file)} — {len(results)} targets, {n} hits\n")
    print(f"{'check':<38}{'pass':>8}{'':>4}{'lost':>6}")
    print("-" * 56)
    print(f"{'1. name appears in evidence':<38}{len(in_ev):>8}/{n:<4}{n-len(in_ev):>6}")
    print(f"{'2. no hallucinated line numbers':<38}{len(no_fake):>8}/{n:<4}{len(in_ev)-len(no_fake):>6}")
    print(f"{'3. a cited line contains the name':<38}{len(named):>8}/{n:<4}{len(no_fake)-len(named):>6}")
    print(f"{'4. cited lines reach the target':<38}{len(conn):>8}/{n:<4}{len(named)-len(conn):>6}")
    print(f"{f'5. bridge admits <={args.max_rivals} other people':<38}{len(uniq):>8}/{n:<4}{len(conn)-len(uniq):>6}")
    print("-" * 56)
    print(f"{'GROUNDED (1-4)':<38}{len(conn):>8}/{n:<4}")
    print(f"{'GROUNDED + DISCRIMINATIVE (1-5)':<38}{len(uniq):>8}/{n:<4}")
    print(f"\nraw accuracy         {n}/{len(results)} = {n/len(results)*100:.1f}%")
    print(f"grounded accuracy    {len(conn)}/{len(results)} = {len(conn)/len(results)*100:.1f}%")
    print(f"discriminative acc.  {len(uniq)}/{len(results)} = {len(uniq)/len(results)*100:.1f}%")

    if conn:
        rv = sorted(c["rivals"] for c in conn)
        print(f"\nrivals per grounded hit: median {rv[len(rv)//2]}, max {rv[-1]}")

    if args.show:
        bad = [c for c in checks if c not in uniq]
        print(f"\n{'='*56}\nfailing cases ({len(bad)}), first {min(args.show, len(bad))}:\n")
        for c in bad[:args.show]:
            reason = ("name never in evidence" if not c["in_evidence"]
                      else f"cited nonexistent lines {c['fake_cited']} (max {c['max_line']})"
                      if c["fake_cited"]
                      else "no cited line contains the name" if not c["name_in_cited"]
                      else "cited lines do not connect name to target" if not c["connected"]
                      else f"bridge also admits {c['rivals']} other named people")
            print(f"  {c['mid']:<14} {c['name']:<24} {reason}")


if __name__ == "__main__":
    main()
