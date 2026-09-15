"""
Step 4: LLM Identity Inference
Verbalize evidence graph → construct attack prompt → LLM predicts real name.
(Adapted from KG-GPT Step 4: Verification → Identity Inference)
"""
import os
import re
import time
from openai import OpenAI

from .step1_target_identification import is_mid
from .config import (
    BASE_URL, MODEL, NVIDIA_API_KEY, SENSITIVE_RELATIONS,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_SEED_KWARGS, TOP_K_PREDICTIONS,
    PROMPTS_DIR, EVIDENCE_FORMAT, LLM_PROVIDER, safe_max_tokens,
    SHOW_NEIGHBOR_PROFILES, LLM_REQUEST_TIMEOUT,
)


_RELATION_VERBS = {
    # Person → [thing]
    "/people/person/employment_history./business/employment_tenure/company": "worked at",
    "/business/job_title/people_with_this_title./business/employment_tenure/company": "held job title at",
    "/people/person/spouse_s./people/marriage/spouse": "was married to",
    "/people/person/sibling_s./people/sibling_relationship/sibling": "is a sibling of",
    "/base/schemastaging/person_extra/net_worth./measurement_unit/dated_money_value/currency": "net worth denominated in",
    "/people/person/place_of_birth": "was born in",
    "/people/person/place_of_death": "died in",
    "/people/person/profession": "has profession",
    "/people/person/nationality": "is a national of",
    "/people/person/religion": "practices religion",
    "/people/person/gender": "gender is",
    "/people/person/education./education/education/institution": "was educated at",
    "/people/person/parents": "has parent",
    "/people/person/children": "has child",
    "/film/actor/film./film/performance/film": "appeared in film",
    "/film/director/film": "directed film",
    "/music/artist/genre": "makes music in genre",
    "/music/artist/album": "released album",
    "/award/award_winner/awards_won./award/award_honor/award": "won award",
    "/award/award_nominee/award_nominations./award/award_nomination/award": "was nominated for award",
    "/organization/organization_founder/organizations_founded": "founded",
    "/sports/pro_athlete/teams./sports/sports_team_roster/team": "played for team",
    # Cause/disease → Person (head=cause, tail=person)
    "/people/cause_of_death/people": "was the cause of death of",
    "/medicine/disease/notable_people_with_this_condition": "is a condition associated with notable person",
    # Other
    "/film/film/directed_by": "was directed by",
    "/film/film/starring./film/performance/actor": "features actor",
    "/location/location/contains": "contains",
}


def clean_relation_name(rel):
    """Return human-readable verb for known Freebase paths; fall back to last path component."""
    base = rel.lstrip("~")
    if base in _RELATION_VERBS:
        verb = _RELATION_VERBS[base]
        return f"~{verb}" if rel.startswith("~") else verb
    # Fall back: take last component(s), replace underscores/dots with spaces
    parts = base.rstrip("/").split("/")
    last = parts[-1].replace("_", " ").replace(".", " ")
    return f"~{last}" if rel.startswith("~") else last


_TAIL_VERBS = {
    # When [TARGET PERSON] is the TAIL of the relation (context_entity is the HEAD):
    # express as "[TARGET PERSON] {tail_verb} {context_entity}"
    "/people/cause_of_death/people": "died of",
    "/medicine/disease/notable_people_with_this_condition": "has/had the condition",
    "/business/job_title/people_with_this_title./business/employment_tenure/company": "held a job at",
}


def _placeholder_label(idx):
    """Map 0 -> [PERSON_A], 1 -> [PERSON_B], ..., wraps after 26 with [PERSON_AA]."""
    if idx < 26:
        return f"[PERSON_{chr(ord('A') + idx)}]"
    a, b = divmod(idx, 26)
    return f"[PERSON_{chr(ord('A') + a - 1)}{chr(ord('A') + b)}]"


def build_placeholder_map(target_mid, evidence_triples, sensitive_contexts,
                           wiki_mapping=None):
    """
    Assign deterministic [PERSON_X] labels to every MID neighbor that is NOT
    the target itself. wiki_mapping is accepted for API compatibility but NOT
    used here — neighbor MIDs must remain anonymous (using their real names
    would leak ground-truth identity information the attacker cannot know).
    """
    seen_order = []
    for trip in evidence_triples:
        for ent in (trip[0], trip[2]):
            if is_mid(ent) and ent != target_mid and ent not in seen_order:
                seen_order.append(ent)
    for ctx in sensitive_contexts:
        ent = ctx["context_entity"]
        if is_mid(ent) and ent != target_mid and ent not in seen_order:
            seen_order.append(ent)

    return {mid: _placeholder_label(i) for i, mid in enumerate(seen_order)}


def _render_entity(ent, target_mid, placeholder_map, wiki_mapping=None):
    if ent == target_mid:
        return "[TARGET PERSON]"
    if ent in placeholder_map:
        return placeholder_map[ent]
    return ent


def summarize_neighbor_profile(KG, mid, max_facts=5,
                                exclude_relations=None, exclude_entities=None,
                                wiki_mapping=None,
                                target_mid=None, placeholder_map=None):
    """
    Build a profile of a placeholder MID for the LLM prompt.

    Two layers:
      1. Non-sensitive public facts (occupation, genre, nationality …) — from
         the real-name KG key, if available (but wiki_mapping NOT used since
         attacker doesn't know the real name).
      2. Sensitive-structure facts (sibling / spouse / employer network) — the
         attacker CAN see these in the anonymized KG.  Endpoints are rendered as
         "[TARGET PERSON]", "[PERSON_X]" (if already in placeholder_map), or
         "[masked person]" for unseen MIDs.  This lets the LLM count, e.g.,
         "PERSON_A has 7 siblings including TARGET" → Jackson family reasoning.

    target_mid + placeholder_map are required for sensitive-structure rendering.
    """
    if KG is None or mid not in KG:
        return []

    exclude_relations = set(exclude_relations) if exclude_relations else set()
    exclude_entities  = set(exclude_entities)  if exclude_entities  else set()
    _pmap             = placeholder_map or {}

    # Layer 1: non-sensitive public facts (string-value tails only)
    public_facts = []
    for rel, tails in KG[mid].items():
        actual_rel = rel.lstrip("~")
        if actual_rel in SENSITIVE_RELATIONS:
            continue
        if actual_rel in exclude_relations:
            continue
        for tail in tails:
            if tail in exclude_entities:
                continue
            if is_mid(tail):
                continue
            arrow = "<-" if rel.startswith("~") else "->"
            public_facts.append(f"{actual_rel} {arrow} {tail}")
            if len(public_facts) >= max_facts:
                break

    # Layer 2: sensitive-structure (who they're connected to, rendered anonymously)
    sensitive_facts = []
    seen_sensitive = set()
    if target_mid is not None:
        for rel, tails in KG[mid].items():
            actual_rel = rel.lstrip("~")
            if actual_rel not in SENSITIVE_RELATIONS:
                continue
            rel_verb = clean_relation_name(actual_rel)
            for tail in tails:
                if tail == target_mid:
                    tail_label = "[TARGET PERSON]"
                elif tail in _pmap:
                    tail_label = _pmap[tail]
                elif is_mid(tail):
                    tail_label = "[masked person]"
                else:
                    tail_label = tail
                fact_str = f"{rel_verb} -> {tail_label}"
                if fact_str not in seen_sensitive:
                    seen_sensitive.add(fact_str)
                    sensitive_facts.append(fact_str)

    return public_facts + sensitive_facts


def verbalize_evidence(target_mid, evidence_triples, wiki_mapping=None,
                       placeholder_map=None):
    """
    Convert structured triples to natural language sentences.

    Rules:
      - target_mid          → "[TARGET PERSON]"
      - Other anonymized MIDs (when placeholder_map is provided) → typed
        placeholder like "[PERSON_A]". This keeps the LLM grounded — it
        treats masked neighbors as known-unknowns rather than free variables
        to hallucinate names for.
      - When placeholder_map is None, falls back to legacy behavior of
        keeping raw MIDs.
      - String entities (already public in graph) → kept as-is.
      - Relations → kept as raw Freebase paths.
    """
    if placeholder_map is None:
        placeholder_map = {}

    _TARGET_TAIL_VERBS = {
        # When [TARGET PERSON] or placeholder is the TAIL, flip sentence order.
        "/people/cause_of_death/people": "died of",
        "/medicine/disease/notable_people_with_this_condition": "has/had the condition",
        "/business/job_title/people_with_this_title./business/employment_tenure/company": "held a job at",
    }
    target_label = "[TARGET PERSON]"
    placeholder_labels = set(placeholder_map.values()) if placeholder_map else set()

    sentences = []
    for trip in evidence_triples:
        h, r, t = trip[0], trip[1], trip[2]
        r_clean = clean_relation_name(r)
        h_disp = _render_entity(h, target_mid, placeholder_map)
        t_disp = _render_entity(t, target_mid, placeholder_map)

        # If target/placeholder is the tail and we have a tail-flip verb, put subject first
        tail_is_person = t_disp == target_label or t_disp in placeholder_labels
        if tail_is_person and r in _TARGET_TAIL_VERBS:
            flip_verb = _TARGET_TAIL_VERBS[r]
            sentences.append(f"{t_disp} {flip_verb} {h_disp}")
        else:
            sentences.append(f"{h_disp} {r_clean} {t_disp}")
    return sentences


def triplify_evidence(target_mid, evidence_triples, wiki_mapping=None,
                      placeholder_map=None):
    """
    Render evidence as raw ``[head, relation, tail]`` triples, KG-GPT style.

    KG-GPT (Kim et al., EMNLP 2023 Findings) never verbalizes: both its FactKG
    and MetaQA prompts pass the evidence as literal triples and explain the
    format once with a single line — "Each evidence is in the form of
    [head, relation, tail] and it means 'head's relation is tail.'".

    Why this matters here rather than being cosmetic: verbalize_evidence()
    shortens the relation to its LAST path component, which collapses
    distinctions this attack depends on. "/people/ethnicity/people" and
    "/location/statistical_region/religions./location/religion_percentage/religion"
    both degrade to a bare word, so a fact about a PERSON's religion becomes
    indistinguishable from a demographic statistic about a STATE — exactly the
    confusion that sends the model from a religion node to Montana/Nevada/Oregon
    instead of to a person. It also flips head/tail for some relations, erasing
    the direction of the edge.

    Keeping the full relation path preserves both the type signature and the
    direction, at no extra token cost worth worrying about.
    """
    if placeholder_map is None:
        placeholder_map = {}

    lines = []
    for trip in evidence_triples:
        h, r, t = trip[0], trip[1], trip[2]
        h_disp = _render_entity(h, target_mid, placeholder_map)
        t_disp = _render_entity(t, target_mid, placeholder_map)
        # Inverse edges are normalised to their forward form so the direction is
        # unambiguous: "~r" between (h, t) is the same fact as r between (t, h).
        if r.startswith("~"):
            h_disp, t_disp, r = t_disp, h_disp, r[1:]
        lines.append(f"['{h_disp}', '{r}', '{t_disp}']")
    return lines


EVIDENCE_FORMAT_NOTE = (
    "Each evidence is in the form of [head, relation, tail] and it means "
    "\"head's relation is tail.\". The relation is the full Freebase path, so its "
    "prefix tells you the TYPE of the subject: /people/person/... is a fact about a "
    "person, while /location/... is a fact about a place (e.g. a population "
    "statistic), not about anyone living there."
)


def render_evidence(target_mid, evidence_triples, wiki_mapping=None,
                    placeholder_map=None, fmt=None):
    """Render evidence in the configured format ("triple" or "sentence")."""
    if fmt is None:
        fmt = EVIDENCE_FORMAT
    renderer = triplify_evidence if fmt == "triple" else verbalize_evidence
    return renderer(target_mid, evidence_triples, wiki_mapping=wiki_mapping,
                    placeholder_map=placeholder_map)


def _build_profile_block(placeholder_map, KG, target_mid, sensitive_contexts,
                          wiki_mapping=None, kg_stats=None):
    """
    Render a "Profile of anonymized placeholders" block with structural
    fingerprint metadata (role, cluster size, anchor neighbors) followed by
    facts (public + sensitive-structure rendered with placeholder labels).

    Gated by config.SHOW_NEIGHBOR_PROFILES — see that flag's docstring for
    which research scenario this represents and why it's a toggle, not a
    fixed default. Every fact rendered here already exists as a raw evidence
    triple (render_evidence); this only pre-groups them by entity, the way a
    real attacker naturally would before analysing a scraped graph.
    """
    if not SHOW_NEIGHBOR_PROFILES:
        return ""
    if not placeholder_map or KG is None:
        return ""

    # Import here to avoid circular dep at module load
    from .step3_evidence_construction import compute_placeholder_fingerprint

    lines = []
    for mid, label in placeholder_map.items():
        fp = compute_placeholder_fingerprint(
            KG, mid, target_mid, placeholder_map=placeholder_map, kg_stats=kg_stats,
        )
        facts = summarize_neighbor_profile(
            KG, mid,
            target_mid=target_mid,
            placeholder_map=placeholder_map,
            wiki_mapping=wiki_mapping,
        )

        lines.append(f"  {label} [{mid}] (anonymized):")

        # Structural fingerprint header
        if fp["role_hints"]:
            lines.append(f"      Role: {', '.join(fp['role_hints'])} of [TARGET PERSON]")
        if fp["cluster_size"] > 0:
            cluster_desc = f"part of a network of {fp['cluster_size']} other anonymous person(s)"
            if fp["sibling_count"] > 0:
                cluster_desc += f" - sibling cluster of {fp['sibling_count']}"
            lines.append(f"      Cluster: {cluster_desc}")
        if fp["anchor_neighbors"]:
            lines.append(f"      Anchor neighbors (named): {', '.join(fp['anchor_neighbors'][:3])}")
        if fp["rarity_score"] >= 0.05:
            lines.append(f"      Discrimination: rare fact present (1/{int(1/fp['rarity_score'])} entities share)")

        # Facts
        if facts:
            lines.append(f"      Facts:")
            for f in facts:
                lines.append(f"        - {f}")
        else:
            lines.append(f"      Facts: (no additional KG facts beyond structural ties)")

    if not lines:
        return ""
    return "Profile of anonymized placeholders:\n" + "\n".join(lines)


def _select_prompt_file(mode, has_candidates):
    """Select prompt template path (relative to PROMPTS_DIR) based on mode and candidate presence."""
    subfolder = "with_candidates" if has_candidates else "free"
    filename   = "open_book.txt" if mode == "open_book" else "closed_book.txt"
    return os.path.join(subfolder, filename)


def construct_attack_prompt(target_mid, evidence_sentences, sensitive_contexts,
                            wiki_mapping=None, mode="open_book",
                            placeholder_map=None, KG=None, candidates=None,
                            kg_stats=None):
    """
    Build de-anonymization prompt.

    Args:
        mode: "open_book"  → LLM được dùng world knowledge + evidence
              "closed_book" → LLM CHỈ được dùng evidence, cấm dùng tri thức nội tại
        placeholder_map: dict mid -> "[PERSON_X]" — when provided, masked
            neighbors in the sensitive context are also rendered with these
            placeholder labels and their public profile is appended so the LLM
            does not hallucinate identities for unknown MIDs.
        KG: dict, used to build placeholder profile section.
        candidates: optional list of person names. When provided, the prompt
            instructs the LLM to pick its top-K answers from this list
            (closed-set evaluation). When None, falls back to open-vocab.
    """
    if placeholder_map is None:
        placeholder_map = {}

    prompt_file = _select_prompt_file(mode, candidates)
    prompt_path = os.path.join(PROMPTS_DIR, prompt_file)
    template = open(prompt_path, 'r', encoding='utf-8').read()

    evidence_text = "\n".join(
        f"  {i+1}. {s}" for i, s in enumerate(evidence_sentences)
    )

    # Sensitive context — position-aware, readable sentences about [TARGET PERSON]
    context_lines = []
    for ctx in sensitive_contexts:
        rel = ctx["relation"]
        ctx_entity = _render_entity(ctx["context_entity"], target_mid, placeholder_map)
        position = ctx.get("position", "tail")
        if position == "tail":
            tail_verb = _TAIL_VERBS.get(rel, clean_relation_name(rel))
            context_lines.append(f"  - [TARGET PERSON] {tail_verb} {ctx_entity}")
        else:
            context_lines.append(f"  - [TARGET PERSON] {clean_relation_name(rel)} {ctx_entity}")
    context_text = "\n".join(context_lines)

    profile_text = _build_profile_block(
        placeholder_map, KG, target_mid, sensitive_contexts,
        wiki_mapping=wiki_mapping, kg_stats=kg_stats,
    )

    # Candidate list block rendered into with_candidates templates
    if candidates:
        candidate_text = "\n".join(f"{j}. {c}" for j, c in enumerate(candidates, 1)) + "\n"
    else:
        candidate_text = ""

    prompt = template.replace("<<<EVIDENCE>>>", evidence_text)
    prompt = prompt.replace("<<<SENSITIVE_CONTEXT>>>", context_text)
    prompt = prompt.replace("<<<PLACEHOLDER_PROFILES>>>", profile_text)
    prompt = prompt.replace("<<<CANDIDATE_LIST>>>", candidate_text)

    return prompt


# ============================================================
# System messages cho từng mode
# ============================================================
SYSTEM_MESSAGES = {
    "open_book": (
        "You are an expert at identifying real-world people from Knowledge Graph evidence. "
        "Strategy: FIRST hypothesize identity for each [PERSON_X] placeholder in "
        "PLACEHOLDER_IDENTITIES, THEN use those hypotheses + 1-2 most distinctive facts "
        "to identify [TARGET PERSON] in PREDICTIONS. "
        "IMPORTANT: Do NOT write any preamble, evidence recap, or reasoning steps. "
        "Your very first output must be 'PLACEHOLDER_IDENTITIES:' — start there."
    ),
    "closed_book": (
        "You identify anonymized people using ONLY the provided evidence — no external knowledge. "
        "Strategy: FIRST hypothesize each [PERSON_X] from evidence facts in PLACEHOLDER_IDENTITIES, "
        "THEN identify [TARGET PERSON] in PREDICTIONS. ALL reasoning ONLY from evidence. "
        "IMPORTANT: Do NOT write any preamble. Your very first output must be 'PLACEHOLDER_IDENTITIES:'."
    ),
    # Iterative closed-book drops the HER step entirely: resolving other MIDs is a
    # measured dead end (their facts are the generic attributes already barred as
    # grounding), so the model goes straight to PREDICTIONS and spends its retrieval
    # budget on NAMED nodes instead. Kept separate so the non-iterative prompts,
    # which still emit PLACEHOLDER_IDENTITIES, are unaffected.
    "closed_book_iterative": (
        "You identify anonymized people from Knowledge Graph evidence. "
        "Every name you predict must APPEAR in the evidence you were shown, with cited facts "
        "linking it to the target. A name you recognise from your own knowledge but cannot find "
        "in the evidence is not a valid answer — say UNKNOWN and request more evidence instead. "
        "Ignore [PERSON_X] placeholders: they are anonymized and can never be named. "
        "Work through the DERIVATION steps A-E first — the name is the output of that "
        "derivation, not a guess you justify afterwards. "
        "IMPORTANT: Do NOT write any preamble. Your very first output must be 'DERIVATION:'."
    ),
}

# Reasoning models (deepseek-v4-flash and similar) sometimes re-derive the same
# step repeatedly in their internal chain of thought and burn the whole token
# budget before ever writing DERIVATION/PREDICTIONS — measured: one captured
# trace held 131,905 characters of reasoning for a prompt that normally needs
# ~20k, with content="" and finish_reason="length". Raising max_tokens only
# buys a longer loop at higher cost, so instead this is appended to the closed-
# book system message ONLY for deepseek: a direct, low-drama instruction to stop
# re-checking and commit to an answer within the existing steps. It is additive
# (the DERIVATION contract from SYSTEM_MESSAGES is unchanged), so grading and
# the fairness checks (name must be IN the evidence) still apply identically.
DEEPSEEK_BUDGET_NOTE = (
    " Work through each DERIVATION step ONCE, in order, and move on — do not "
    "re-verify a step you already completed or restate the evidence before "
    "committing to PREDICTIONS. If step B's candidate list is inconclusive after "
    "one pass, say so in step D and move to RETRIEVAL_REQUEST rather than "
    "re-scanning the evidence again in the same turn."
)
if LLM_PROVIDER == "deepseek":
    SYSTEM_MESSAGES["closed_book_iterative"] += DEEPSEEK_BUDGET_NOTE


# ============================================================
# Reasoning-mode plumbing, per provider
# ============================================================
# The qwen provider (self-hosted proxy, see API_USAGE (1).md) has an internal
# reasoning pass, toggled via a request-level flag, that is SEPARATE from the
# DERIVATION A-E write-up the prompt already demands in `content` — that
# write-up is a fixed output format the system prompt enforces regardless of
# this flag, not a byproduct of it. Left on, qwen's hidden reasoning pass
# balloons `message.reasoning` and, worse, the accumulated iterative-round
# conversation history, blowing well past this model's 8192-token context
# within 2-3 rounds (measured: round 1 alone hit 9866 input tokens with
# thinking on). Keep it OFF so the token budget stays governed by
# MAX_EVIDENCE_TRIPLES/N_REFINEMENT_ROUNDS the way it is for every other
# provider — the DERIVATION explanation this pipeline actually scores is
# unaffected either way.
_THINKING_EXTRA_BODY = {}  # no provider currently requests reasoning mode


def _llm_call_kwargs():
    """extra_body (if any) to request reasoning mode from the active provider."""
    extra = _THINKING_EXTRA_BODY.get(LLM_PROVIDER)
    return {"extra_body": extra} if extra else {}


def _extract_reasoning(message):
    """Chain-of-thought text, whichever field this provider puts it in."""
    return (getattr(message, "reasoning", "") or
            getattr(message, "reasoning_content", "") or "")


def parse_recognition_check(response_text):
    """Read DERIVATION step E — did the model recognise the person beforehand?

    Recognition cannot be prevented in an LLM that has read the web, and forbidding
    it only teaches the model to hide it. The prompt therefore asks for it openly and
    states it carries no penalty, which turns an unmeasurable confound into a field:
    hits can be split by whether prior knowledge was involved.

    Returns "YES" / "NO" / None (step absent).
    """
    if not response_text:
        return None
    m = re.search(r"E\.\s*RECOGNITION\s*CHECK[^\n]*?\b(YES|NO)\b", response_text, re.I)
    if not m:
        m = re.search(r"RECOGNITION\s*CHECK[^\n]*?\b(YES|NO)\b", response_text, re.I)
    return m.group(1).upper() if m else None


def parse_top1_confidence(response_text):
    """
    Extract the confidence percentage of the LLM's top-1 prediction.
    Looks for patterns like "1. Name (confidence: 85%)" on the first
    numbered line. Returns float in [0, 100], or None if not found.
    """
    if not response_text:
        return None
    for line in response_text.split('\n'):
        line = line.strip()
        # Strip PREDICTIONS: prefix (LLM sometimes puts "PREDICTIONS: 1. Name" on one line)
        line = re.sub(r'^(?:PREDICTIONS|PREDICTION)[:\s]+', '', line, flags=re.IGNORECASE)
        if not line.startswith("1."):
            continue
        m = re.search(r"(?:confidence[:\s]*)(\d+(?:\.\d+)?)\s*%", line, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None
    return None


def parse_placeholder_identities(response_text, placeholder_map):
    """
    Parse the PLACEHOLDER_IDENTITIES section from LLM response.

    Expected format:
      PLACEHOLDER_IDENTITIES:
      - [PERSON_A] = La Toya Jackson (confidence: 75%) — rationale here
      - [PERSON_B] = UNKNOWN (confidence: 0%) — insufficient evidence

    Returns:
        dict {mid: {"identity": str, "confidence": float, "rationale": str}}
        Keys are MIDs (resolved from placeholder labels via placeholder_map).
        Confidence is in [0, 100]. identity may be "UNKNOWN".
    """
    out = {}
    if not response_text or not placeholder_map:
        return out

    # Reverse map: "[person_a]" -> "/m/EXAMPLE01"
    label_to_mid = {label.lower(): mid for mid, label in placeholder_map.items()}

    # Locate section: everything between PLACEHOLDER_IDENTITIES and next ALL_CAPS_HEADER
    section_match = re.search(
        r'PLACEHOLDER_IDENTITIES\s*:\s*(.+?)(?=\n\s*(?:PREDICTIONS|RETRIEVAL_REQUEST|=== )|\Z)',
        response_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not section_match:
        return out
    section = section_match.group(1)

    # Parse each line: "- [PERSON_A] = Name (confidence: NN%) — rationale"
    # or "- [PERSON_A] = UNKNOWN (confidence: 0%) — ..."
    line_re = re.compile(
        r'\[\s*(PERSON_[A-Z]+)\s*\]\s*=\s*(.+?)\s*\(confidence\s*[:\s]\s*(\d+(?:\.\d+)?)\s*%\s*\)\s*(?:[—\-:]\s*(.*))?$',
        flags=re.IGNORECASE,
    )

    for line in section.splitlines():
        line = line.strip().lstrip('-*•').strip()
        if not line:
            continue
        m = line_re.search(line)
        if not m:
            continue
        label = f"[{m.group(1).upper()}]"
        identity = m.group(2).strip().strip('"\'')
        try:
            conf = float(m.group(3))
        except ValueError:
            conf = 0.0
        rationale = (m.group(4) or "").strip().lstrip("-—–:").strip()

        mid = label_to_mid.get(label.lower())
        if mid is None:
            continue
        out[mid] = {
            "identity": identity,
            "confidence": conf,
            "rationale": rationale,
        }
    return out


def parse_predictions(response_text, candidates=None):
    """
    Parse LLM response to extract predicted names.
    Expected format:
      1. Al Pacino (confidence: 95%)
      2. Robert De Niro (confidence: 3%)
      3. Marlon Brando (confidence: 2%)

    When `candidates` is provided, predictions are filtered to names that
    appear in the candidate list (case-insensitive, whitespace-tolerant).
    Predictions outside the candidate set are dropped — this prevents the
    LLM from cheating closed-set evaluation by inventing variants like
    "Christopher Wallace" when only "The Notorious B.I.G." is in the list.
    """
    predictions = []
    if not response_text:
        return predictions

    candidate_lookup = None
    if candidates:
        candidate_lookup = {c.lower().strip(): c for c in candidates}

    # Restrict parsing to PREDICTIONS section if present — avoids accidentally
    # picking up PLACEHOLDER_IDENTITIES lines (which also have "confidence: N%").
    pred_section_match = re.search(
        r'PREDICTIONS\s*:\s*(.+?)(?=\n\s*(?:RETRIEVAL_REQUEST|=== |\Z))',
        response_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    parse_text = pred_section_match.group(1) if pred_section_match else response_text

    for line in parse_text.split('\n'):
        line = line.strip()
        # Handle "PREDICTIONS: 1. Name" on the same line (LLM formatting quirk)
        line = re.sub(r'^(?:PREDICTIONS|PREDICTION)[:\s]+', '', line, flags=re.IGNORECASE)
        # Require "(confidence:" to distinguish prediction lines from evidence
        # items — LLMs sometimes number evidence facts (e.g. "1. [TARGET PERSON]
        # has dyslexia") which would otherwise be falsely parsed as predictions.
        if '(confidence:' not in line.lower():
            continue
        match = re.match(r'^\d+\.\s*(.+?)(?:\s*\(confidence)', line, re.IGNORECASE)
        if not match:
            continue
        name = match.group(1).strip()
        if not name or len(name) <= 1 or name.upper() == "UNKNOWN":
            continue
        if candidate_lookup is not None:
            key = name.lower().strip()
            canonical = candidate_lookup.get(key)
            if canonical is None:
                continue
            predictions.append(canonical)
        else:
            predictions.append(name)
    return predictions


def infer_identity(target_mid, evidence_triples, sensitive_contexts,
                   wiki_mapping=None, api_key=None, mode="open_book",
                   KG=None, candidates=None):
    """
    Full Step 4: Verbalize → Prompt → LLM → Parse predictions.

    Args:
        mode: "open_book"  → LLM + World Knowledge
              "closed_book" → LLM chỉ dùng evidence
        KG: optional adjacency dict; when provided, masked-neighbor MIDs are
            rendered as typed placeholders and a public-profile block is
            appended to the prompt to prevent the LLM from hallucinating
            identities for unknown MIDs.

    Returns:
        predictions: list of predicted names (top-K)
        raw_response: str, full LLM response
        prompt: str, the constructed prompt
    """
    if api_key is None:
        api_key = NVIDIA_API_KEY

    placeholder_map = build_placeholder_map(target_mid, evidence_triples, sensitive_contexts)

    # 4a. Render evidence (raw triples by default, KG-GPT style)
    sentences = render_evidence(
        target_mid, evidence_triples, wiki_mapping,
        placeholder_map=placeholder_map,
    )

    # 4b. Build prompt (theo mode)
    prompt = construct_attack_prompt(
        target_mid, sentences, sensitive_contexts, wiki_mapping, mode=mode,
        placeholder_map=placeholder_map, KG=KG, candidates=candidates,
    )
    
    # 4c. Call LLM (system message khác nhau theo mode)
    client = OpenAI(base_url=BASE_URL, api_key=api_key, timeout=LLM_REQUEST_TIMEOUT)
    system_msg = SYSTEM_MESSAGES.get(mode, SYSTEM_MESSAGES["open_book"])
    
    user_content = "/no_think\n" + prompt

    call_messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]

    raw_response = ""
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
                **_llm_call_kwargs(),
            )
            _elapsed = time.time() - _t0
            _prompt_chars = sum(len(m.get("content", "")) for m in call_messages)
            print(f"  [TIMING] open_book call: {_elapsed:.1f}s  "
                  f"(prompt ~{_prompt_chars} chars, ~{_prompt_chars//3} tok)")
            raw_response = response.choices[0].message.content or ""
            break
        except Exception as e:
            print(f"  [LLM ERROR attempt {attempt+1}] {e}")
            time.sleep(2)

    # 4d. Parse predictions (filter to candidates if provided)
    predictions = parse_predictions(raw_response, candidates=candidates)
    placeholder_identities = parse_placeholder_identities(raw_response, placeholder_map)

    return predictions, raw_response, prompt, placeholder_identities


# ============================================================
# Iterative Refinement Functions
# ============================================================

def construct_iterative_prompt(target_mid, evidence_sentences, sensitive_contexts,
                                round_num, max_rounds, previous_predictions=None,
                                mode="open_book", placeholder_map=None, KG=None,
                                candidates=None, wiki_mapping=None,
                                previous_placeholder_identities=None,
                                kg_stats=None, failed_requests=None):
    """
    Build prompt cho iterative rounds (Round 1 → N-1).
    LLM vừa dự đoán, vừa đưa ra RETRIEVAL_REQUEST cho round tiếp theo.

    failed_requests: list of human-readable request strings that were already
        tried and returned NOTHING. Fed back so the model stops repeating dead
        ends and redirects its exploration elsewhere.
    """
    if placeholder_map is None:
        placeholder_map = {}

    if mode == "closed_book":
        # Every provider uses the same full prompt (3 worked examples,
        # mandatory DERIVATION A-E, the retrieval strategy that cut
        # mistargeted requests from 66% to 0%, the GROUNDING requirement) so
        # cross-model results are directly comparable. qwen previously needed
        # a trimmed compact/free variant under its old 8192-token TOTAL
        # context cap; now that its context matches the other providers, it
        # no longer needs the trim. inference_iterative_closed_book_compact.txt
        # / _free.txt are kept in the prompts dir for that smaller-context
        # case, not used by default.
        prompt_file = "inference_iterative_closed_book.txt"
    else:
        prompt_file = "inference_iterative.txt"
    prompt_path = os.path.join(PROMPTS_DIR, prompt_file)
    template = open(prompt_path, 'r', encoding='utf-8').read()

    evidence_text = "\n".join(
        f"  {i+1}. {s}" for i, s in enumerate(evidence_sentences)
    )

    context_lines = []
    for ctx in sensitive_contexts:
        rel = ctx["relation"]
        ctx_entity = _render_entity(ctx["context_entity"], target_mid, placeholder_map)
        position = ctx.get("position", "tail")
        if position == "tail":
            tail_verb = _TAIL_VERBS.get(rel, clean_relation_name(rel))
            context_lines.append(f"  - [TARGET PERSON] {tail_verb} {ctx_entity}")
        else:
            context_lines.append(f"  - [TARGET PERSON] {clean_relation_name(rel)} {ctx_entity}")
    context_text = "\n".join(context_lines)

    profile_text = _build_profile_block(
        placeholder_map, KG, target_mid, sensitive_contexts,
        wiki_mapping=wiki_mapping, kg_stats=kg_stats,
    )

    prev_text = ""
    if previous_predictions and round_num > 1:
        prev_text = "=== PREVIOUS ROUND HYPOTHESES (review critically) ===\n"
        for i, pred in enumerate(previous_predictions, 1):
            prev_text += f"  {i}. {pred}\n"
        prev_text += "\nReview these hypotheses against the new evidence. Update or discard if new facts contradict them.\n\n"

    ph_prev_text = ""
    if previous_placeholder_identities and round_num > 1:
        rows = []
        for mid, info in previous_placeholder_identities.items():
            if mid not in placeholder_map:
                continue
            label = placeholder_map[mid]
            ident = info.get("identity", "UNKNOWN")
            conf  = info.get("confidence", 0.0)
            rows.append(f"  {label} last guess: {ident} ({conf:.0f}%) — reaffirm or revise")
        if rows:
            ph_prev_text = ("=== PREVIOUS PLACEHOLDER HYPOTHESES (review critically) ===\n"
                            + "\n".join(rows) + "\n\n")

    if candidates:
        cand_lines = ["=== CANDIDATE SUGGESTIONS (reference only — not exhaustive) ==="]
        for j, c in enumerate(candidates, 1):
            cand_lines.append(f"{j}. {c}")
        cand_lines.append(
            "\nUse this list as a starting point. Pick from it if evidence matches, "
            "or name someone outside the list if evidence clearly points elsewhere.\n"
        )
        candidate_text = "\n".join(cand_lines) + "\n"
    else:
        candidate_text = ""

    # Dead-end feedback: tell the model which requests already came back empty so
    # it redirects instead of re-asking the same thing every round.
    failed_text = ""
    if failed_requests:
        failed_text = (
            "=== REQUESTS ALREADY TRIED THAT RETURNED NOTHING ===\n"
            + "\n".join(f"  - {f}" for f in failed_requests)
            + "\n\nThe graph has no such facts for those nodes. Do NOT repeat these requests.\n"
              "Redirect your exploration to a DIFFERENT node or a DIFFERENT relation.\n\n"
        )

    prompt = template.replace("<<<TARGET_MID>>>", target_mid)
    prompt = prompt.replace("<<<EVIDENCE>>>", evidence_text)
    prompt = prompt.replace("<<<SENSITIVE_CONTEXT>>>", context_text)
    prompt = prompt.replace("<<<PLACEHOLDER_PROFILES>>>", profile_text)
    prompt = prompt.replace("<<<ROUND>>>", str(round_num))
    prompt = prompt.replace("<<<MAX_ROUNDS>>>", str(max_rounds))
    prompt = prompt.replace("<<<PREVIOUS_PREDICTIONS>>>", prev_text + failed_text)
    prompt = prompt.replace("<<<PLACEHOLDER_HYPOTHESES_PREVIOUS>>>", ph_prev_text)
    prompt = prompt.replace("<<<CANDIDATE_LIST>>>", candidate_text)

    return prompt


def parse_retrieval_request(response_text, placeholder_map=None, target_mid=None,
                             evidence_entities=None):
    """
    Parse RETRIEVAL_REQUEST section from LLM response.

    Two request kinds are supported:

    1. Whole-neighbour exploration (pull everything about one node):
         EXPLORE: [PERSON_A]        ← preferred (placeholder label)
         IDENTIFY: [PERSON_A]       ← alias, same effect
         EXPLORE: /m/EXAMPLE02          ← legacy raw MID

    2. Targeted relation pull (pull ONLY one relation of one node — precise,
       low-noise, and a much stronger signal that the model reasoned about
       *what kind* of fact it needs):
         NEED_RELATION: <relation-substring> of [PERSON_A]
         NEED_RELATION: <relation-substring> of TARGET
         NEED_RELATION: <relation-substring> of <Named Entity In Evidence>

    Fairness note: the node must already appear in the current evidence (the
    orchestrator enforces this), so the model can only drill into what it has
    actually observed — it cannot conjure an entity from outside knowledge.

    Args:
        placeholder_map: {mid: "[PERSON_X]"} to resolve labels back to MIDs.
        target_mid: resolves the literal word TARGET / [TARGET PERSON].
        evidence_entities: optional set of entity strings currently in evidence,
            used to resolve NEED_RELATION requests naming a plain string entity.

    Returns:
        dict with:
          "entities":      list of node ids to expand fully
          "relation_reqs": list of (node_id, relation_substring) to pull precisely
    """
    requests = {"entities": [], "relations": [], "keywords": [], "relation_reqs": []}

    if not response_text:
        return requests

    # Build reverse map: "[person_a]" -> "/m/EXAMPLE02"
    label_to_mid = {}
    if placeholder_map:
        for mid, label in placeholder_map.items():
            label_to_mid[label.lower()] = mid

    # Case-insensitive lookup for named entities present in the evidence
    ev_lookup = {}
    if evidence_entities:
        for e in evidence_entities:
            ev_lookup[e.lower()] = e

    # Find the RETRIEVAL_REQUEST section (everything after the marker to the end)
    retrieval_section_match = re.search(
        r'RETRIEVAL_REQUEST\s*[:\s-]*(.+)',
        response_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not retrieval_section_match:
        return requests

    retrieval_text = retrieval_section_match.group(1)

    def _resolve_node(raw):
        """Map a written node reference onto a real KG key, or None."""
        v = raw.strip().strip('"\'').rstrip(',;.').strip()
        if not v:
            return None
        vl = v.lower()
        if vl in label_to_mid:
            return label_to_mid[vl]
        if target_mid and ("target" in vl):
            return target_mid
        if v.startswith('/m/'):
            return v
        if vl in ev_lookup:          # a named entity that IS in the evidence
            return ev_lookup[vl]
        return None

    # --- 1. EXPLORE / IDENTIFY: whole-node expansion ---
    for val in re.findall(r'(?:EXPLORE|IDENTIFY)\s*:\s*(\[?[^\n,;]+?\]?)(?=\s*(?:[-—–]|\n|$))',
                          retrieval_text, flags=re.IGNORECASE):
        node = _resolve_node(val)
        if node and node not in requests["entities"]:
            requests["entities"].append(node)

    # --- 2. NEED_RELATION: <relation> of <node> ---
    for rel, who in re.findall(
            r'NEED_RELATION\s*:\s*(\S+)\s+of\s+(\[?[^\n,;]+?\]?)(?=\s*(?:[-—–]|\n|$))',
            retrieval_text, flags=re.IGNORECASE):
        rel = rel.strip().strip('"\'').rstrip(',;.')
        node = _resolve_node(who)
        if node and rel:
            pair = (node, rel)
            if pair not in requests["relation_reqs"]:
                requests["relation_reqs"].append(pair)

    return requests


def infer_with_feedback(target_mid, evidence_triples, sensitive_contexts,
                         round_num, max_rounds, previous_predictions=None,
                         api_key=None, mode="open_book", KG=None,
                         candidates=None, wiki_mapping=None,
                         previous_placeholder_identities=None,
                         kg_stats=None, failed_requests=None,
                         conversation=None, new_triples=None):
    """
    Iterative inference: LLM dự đoán + đưa ra retrieval requests.

    Multi-turn mode (Cách B): when `conversation` is supplied, the model sees its
    OWN previous turns as real chat history, so it remembers *why* it asked for
    each piece of evidence and can carry a multi-step plan across rounds
    (resolve [PERSON_A] -> use A to pin down TARGET). Round 1 sends the full
    prompt; later rounds send only a short delta message listing the newly
    retrieved facts, which keeps the cached prefix stable.

    Args:
        round_num: round hiện tại (1-indexed)
        max_rounds: tổng số rounds
        previous_predictions: list of predictions từ round trước
        KG: optional adjacency dict to enable typed-placeholder profiles.
        wiki_mapping: dict MID -> real_name; when provided, non-target neighbor
            MIDs are shown as real names in evidence sentences instead of opaque
            [PERSON_X] placeholders, dramatically improving LLM disambiguation.
        conversation: list of {"role","content"} messages carried across rounds.
            Mutated in place (the new user turn and assistant reply are appended)
            so the caller can simply pass the same list every round.
        new_triples: triples retrieved since the previous round, rendered as
            [NEW] lines so the model can see exactly what its request returned.

    Returns:
        predictions, retrieval_requests, raw_response, prompt,
        top1_confidence, placeholder_identities
    """
    if api_key is None:
        api_key = NVIDIA_API_KEY

    placeholder_map = build_placeholder_map(
        target_mid, evidence_triples, sensitive_contexts, wiki_mapping=wiki_mapping,
    )
    sentences = render_evidence(
        target_mid, evidence_triples, wiki_mapping=wiki_mapping,
        placeholder_map=placeholder_map,
    )

    multi_turn = conversation is not None
    is_first_turn = (not multi_turn) or (len(conversation) == 0)

    if round_num >= max_rounds:
        prompt = construct_attack_prompt(
            target_mid, sentences, sensitive_contexts, wiki_mapping=wiki_mapping,
            mode=mode, placeholder_map=placeholder_map, KG=KG, candidates=candidates,
            kg_stats=kg_stats,
        )
    else:
        prompt = construct_iterative_prompt(
            target_mid, sentences, sensitive_contexts,
            round_num, max_rounds, previous_predictions, mode=mode,
            placeholder_map=placeholder_map, KG=KG, candidates=candidates,
            wiki_mapping=wiki_mapping,
            previous_placeholder_identities=previous_placeholder_identities,
            kg_stats=kg_stats, failed_requests=failed_requests,
        )

    # ─── Build the message list ────────────────────────────────────────
    # The iterative closed-book prompt no longer emits PLACEHOLDER_IDENTITIES, so it
    # needs its own system message. The final round falls back to the non-iterative
    # attack prompt, which still does — keep the original message there.
    uses_iterative_prompt = (round_num < max_rounds)
    if mode == "closed_book" and uses_iterative_prompt:
        system_msg = SYSTEM_MESSAGES["closed_book_iterative"]
    else:
        system_msg = SYSTEM_MESSAGES.get(mode, SYSTEM_MESSAGES["open_book"])

    if multi_turn and not is_first_turn:
        # Later rounds: the model already holds the full task + earlier evidence
        # in its own history, so send only what changed. This preserves its
        # stated intent ("I asked for A's films because...") for free.
        new_lines = render_evidence(
            target_mid, new_triples or [], wiki_mapping=wiki_mapping,
            placeholder_map=placeholder_map,
        )
        if new_lines:
            # Number the new facts, continuing from the evidence the model already
            # holds. Without numbers it cannot cite what its own request returned:
            # the prompt demands "GROUNDING: facts <numbers>", so an unnumbered
            # [NEW] line forces it to either invent a number or drop the fact.
            # Measured before this fix: hits citing lines 203-260 when only 200
            # existed — the facts were real, the numbers were not.
            first_new = len(evidence_triples) - len(new_lines) + 1
            delta = ("Your requests returned these NEW facts "
                     f"(round {round_num} of {max_rounds}):\n"
                     + "\n".join(f"  [NEW] {first_new + i}. {s}"
                                 for i, s in enumerate(new_lines)))
        else:
            delta = (f"Your previous requests returned NO new facts "
                     f"(round {round_num} of {max_rounds}). "
                     "Those nodes/relations are exhausted — redirect to a different one.")

        if failed_requests:
            delta += ("\n\nAlready tried and returned nothing (do not repeat):\n"
                      + "\n".join(f"  - {f}" for f in failed_requests))

        her = not (mode == "closed_book" and uses_iterative_prompt)
        # The iterative closed-book prompt answers through a DERIVATION block (A-E)
        # that must be redone each round: new facts can add candidates to step B or
        # supply the eliminating fact step D was missing.
        sections = ("PLACEHOLDER_IDENTITIES and PREDICTIONS" if her
                    else "DERIVATION (A-E) and PREDICTIONS")
        if round_num >= max_rounds:
            delta += (f"\n\nThis is the FINAL round — no more retrieval is possible. "
                      f"Give your best {sections} now, applying the admissibility rule.")
        else:
            delta += (f"\n\nUpdate {sections} in light of these, "
                      "then issue your next RETRIEVAL_REQUEST (same format as before). "
                      "Remember: requests aimed at [PERSON_X] or /m/... nodes are rejected — "
                      "ask about NAMED entities only.")

        # "/no_think" was only prepended on the FIRST turn (see the else branch below);
        # every later round sent the delta with no such hint, so a reasoning model got
        # 4 unguided rounds after 1 guided one — and reasoning length grows with
        # accumulated conversation history, which is exactly where it was most needed.
        # Harmless if the backend ignores it (plain text prefix), so apply it uniformly.
        user_content = "/no_think\n" + delta
        # Saved turns may carry an extra "reasoning" key (a reasoning model's chain
        # of thought, kept for analysis). Strip it here: the API rejects unknown
        # message fields, and re-sending the thinking would bloat every later round.
        messages = ([{"role": "system", "content": system_msg}]
                    + [{"role": m["role"], "content": m["content"]} for m in conversation]
                    + [{"role": "user", "content": user_content}])
    else:
        user_content = "/no_think\n" + prompt
        messages = [{"role": "system", "content": system_msg},
                    {"role": "user", "content": user_content}]

    # Call LLM
    client = OpenAI(base_url=BASE_URL, api_key=api_key, timeout=LLM_REQUEST_TIMEOUT)

    raw_response = ""
    reasoning_content = ""
    for attempt in range(3):
        try:
            _t0 = time.time()
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=safe_max_tokens(messages),
                temperature=LLM_TEMPERATURE,
                top_p=0.1,
                **LLM_SEED_KWARGS,
                **_llm_call_kwargs(),
            )
            _elapsed = time.time() - _t0
            _prompt_chars = sum(len(m.get("content", "")) for m in messages)
            _usage = getattr(response, "usage", None)
            _usage_str = (f"prompt_tokens={_usage.prompt_tokens} "
                          f"completion_tokens={_usage.completion_tokens}"
                          if _usage else "usage=n/a")
            print(f"  [TIMING] closed_book call: {_elapsed:.1f}s  "
                  f"(prompt ~{_prompt_chars} chars, ~{_prompt_chars//3} tok est.  {_usage_str})")
            choice = response.choices[0]
            raw_response = choice.message.content or ""
            # Reasoning models put their chain of thought in a separate field and
            # only the final answer in `content`. Kept for analysis — this experiment
            # measures HOW a name was reached, not just which name.
            reasoning_content = _extract_reasoning(choice.message)
            if raw_response.strip():
                break
            # Empty content with finish_reason="length" means the model looped: it
            # rewrote the same paragraph until it hit the cap without ever emitting
            # an answer (measured: 618 sentences, 100 distinct). It is intermittent
            # — the same input succeeds on most calls — so just retry.
            if choice.finish_reason == "length":
                print(f"  [LLM LOOPED attempt {attempt+1}] burned {LLM_MAX_TOKENS} "
                      f"tokens with no answer — retrying")
            else:
                print(f"  [LLM EMPTY attempt {attempt+1}] "
                      f"finish_reason={choice.finish_reason} — retrying")
            reasoning_content = ""   # degenerate loop text: not worth keeping
            time.sleep(2)
        except Exception as e:
            print(f"  [LLM ERROR attempt {attempt+1}] {e}")
            time.sleep(2)

    if not raw_response.strip():
        print(f"  [LLM] no answer after 3 attempts — round recorded as empty")

    # Grow the history so the next round sees this exchange verbatim. Only the
    # answer goes back to the model — re-sending the chain of thought would bloat
    # every later round's context for no benefit. The reasoning is kept separately
    # (see below) because this experiment measures HOW a name was reached.
    if multi_turn:
        conversation.append({"role": "user", "content": user_content})
        conversation.append({"role": "assistant", "content": raw_response,
                             **({"reasoning": reasoning_content}
                                if reasoning_content.strip() else {})})
    
    predictions = parse_predictions(raw_response, candidates=candidates)
    # Entities currently visible in the evidence — NEED_RELATION may only target
    # one of these (fairness: the model can drill only into what it has seen).
    _ev_entities = {e for trip in evidence_triples for e in (trip[0], trip[2])}

    # Closed-book means the answer must come FROM the graph. The prompt says so, but
    # nothing enforced it: measured on /m/EXAMPLE03, gemini cited six real facts — all
    # of them about the target ("executive produced Scary Movie 2, The Cider House
    # Rules, Iris") — and then supplied the producer's name from memory, because no
    # evidence line contained that name. Every citation was valid; the identification
    # still came from outside the graph.
    #
    # Enforce the missing half here: a predicted name has to APPEAR in the evidence
    # the model was shown. This does not police HOW the name is used (verify_grounding.py
    # checks that afterwards), but it removes the case where the graph never named the
    # person at all — which is memory, not inference, by definition.
    if mode == "closed_book":
        _visible = {e.lower() for e in _ev_entities if isinstance(e, str)}
        _kept = []
        for _name in predictions:
            if _name.lower() in _visible:
                _kept.append(_name)
            else:
                print(f"  [CLOSED-BOOK] dropped '{_name}' — name is not in the evidence")
        predictions = _kept
    retrieval_requests = parse_retrieval_request(
        raw_response, placeholder_map=placeholder_map,
        target_mid=target_mid, evidence_entities=_ev_entities,
    )
    top1_confidence = parse_top1_confidence(raw_response)
    placeholder_identities = parse_placeholder_identities(raw_response, placeholder_map)

    return predictions, retrieval_requests, raw_response, prompt, top1_confidence, placeholder_identities

