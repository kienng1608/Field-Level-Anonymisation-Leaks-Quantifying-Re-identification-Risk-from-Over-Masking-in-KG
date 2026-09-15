"""
Configuration for the De-Anonymization Attack Pipeline.
"""
import os
from dotenv import load_dotenv

# Tự động đọc key từ file .env
load_dotenv()

# ============================================================
# LLM API Configuration
# ============================================================
# Provider is selectable so the same pipeline can hit different backends.
# Set LLM_PROVIDER in .env (or the shell) to switch; each provider carries
# its own key, base URL and model id.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "xah").lower()

_PROVIDERS = {
    # DeepSeek platform, direct. Reasoning model: returns chain of thought in
    # `reasoning_content`, answer in `content`.
    "deepseek": {
        "key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com",
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    },
    # xah.io proxy (Gemini under a proxy name). Used for the 210-target runs.
    "xah": {
        "key": os.getenv("XAH_API_KEY"),
        "base_url": "https://api.xah.io/v1",
        "model": os.getenv("XAH_MODEL", "levuphong2909/gemini-3.5-flash-high"),
    },
    # Self-hosted FastAPI proxy in front of vLLM, per API_USAGE (1).md.
    # OpenAI-compatible. Called with thinking OFF (see _THINKING_EXTRA_BODY in
    # step4_llm_inference.py). Context (vLLM --max-model-len) has been raised
    # to match Gemini/DeepSeek's headroom — see LLM_MAX_TOKENS below.
    "qwen": {
        "key": os.getenv("QWEN_API_KEY"),
        "base_url": "http://171.226.10.154:8080/v1",
        "model": os.getenv("QWEN_MODEL", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"),
    },
    # Google AI Studio, direct (OpenAI-compatible endpoint). Large context
    # window like DeepSeek/xah — none of the qwen-specific shrinking applies.
    "gemini": {
        "key": os.getenv("GEMINI_API_KEY"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    },
}
if LLM_PROVIDER not in _PROVIDERS:
    raise ValueError(f"LLM_PROVIDER={LLM_PROVIDER!r}; expected one of {sorted(_PROVIDERS)}")
_P = _PROVIDERS[LLM_PROVIDER]

# Name kept as NVIDIA_API_KEY because the rest of the pipeline imports it under
# that name; it now holds whichever provider's key is selected.
NVIDIA_API_KEY = _P["key"]
BASE_URL = _P["base_url"]
MODEL = os.getenv("MODEL_OVERRIDE") or _P["model"]

# ============================================================
# 7 Sensitive Relations (Dynamic Masking targets)
# Selected by privacy principle: GDPR Art. 9 special categories
# (health, sexual life, religion, ethnicity, political opinion) + financial PII.
# Entities "Người" trong các relation này bị giữ dạng MID.
# Must stay identical to SENSITIVE_RELATIONS in codes/create_anonymized_v2.py.
# ============================================================
SENSITIVE_RELATIONS = [
    "/medicine/disease/notable_people_with_this_condition",                                       # sức khỏe/bệnh tật
    "/people/cause_of_death/people",                                                              # nguyên nhân tử vong
    "/celebrities/celebrity/sexual_relationships./celebrities/romantic_relationship/celebrity",   # đời sống tình dục
    "/people/person/religion",                                                                    # tôn giáo
    "/people/ethnicity/people",                                                                   # dân tộc/chủng tộc
    "/government/political_party/politicians_in_this_party./government/political_party_tenure/politician",  # quan điểm chính trị
    "/base/schemastaging/person_extra/net_worth./measurement_unit/dated_money_value/currency",    # tài chính cá nhân
]

# Which endpoint (head / tail / both) holds the anonymized PERSON in each
# sensitive relation. Step 1 uses this so MIDs of NON-person endpoints (a
# religion, a currency, an ethnicity node) are not mistaken for de-anon targets.
# Must stay in sync with codes/create_anonymized_v2.py.
SENSITIVE_VICTIM_POSITION = {
    "/medicine/disease/notable_people_with_this_condition":                                       "tail",
    "/people/cause_of_death/people":                                                              "tail",
    "/celebrities/celebrity/sexual_relationships./celebrities/romantic_relationship/celebrity":   "both",
    "/people/person/religion":                                                                    "head",
    "/people/ethnicity/people":                                                                   "tail",
    "/government/political_party/politicians_in_this_party./government/political_party_tenure/politician": "tail",
    "/base/schemastaging/person_extra/net_worth./measurement_unit/dated_money_value/currency":    "head",
}

# ============================================================
# Data Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANON_DATA_DIR = os.path.join(BASE_DIR, "data", os.getenv("ANON_DS", "FB15k-237-id-move05"))
ORIG_DATA_DIR = os.path.join(BASE_DIR, "data", "FB15k-237")
WIKI_MAPPING_PATH = os.path.join(ORIG_DATA_DIR, "fb_wiki_mapping.tsv")
PROMPTS_DIR = os.path.join(BASE_DIR, "deanon_pipeline", "prompts")
RESULTS_DIR = os.path.join(BASE_DIR, "deanon_results")

# ============================================================
# Pipeline Parameters
# ============================================================
TOP_K_RELATIONS = 15        # Số relations LLM chọn ở Step 2 (10 -> 15)

# Evidence subgraph cap. 150: measured that hits commonly need 200+ evidence
# lines before the deciding fact appears, so this stays as close to that as
# the context budget allows. qwen's context was previously a hard 8192-token
# ceiling that forced a much smaller cap plus a trimmed prompt; now that its
# context matches the other providers, it uses the same value and the same
# full prompt (inference_iterative_closed_book.txt) so all providers run
# under identical conditions for cross-model comparison.
MAX_EVIDENCE_TRIPLES = 150

# Số hop cho multi-hop expansion. 3 is where most real names sit relative to
# the MID; going higher adds no recall while making retrieval much slower.
N_HOPS = 5
TOP_K_PREDICTIONS = 5       # Số predictions LLM trả về ở Step 4
# temperature=0 + a fixed seed minimizes (but does not eliminate) run-to-run
# variance: commercial APIs generally treat `seed` as best-effort, not a
# bit-exact reproducibility guarantee (unlike a locally-hosted model run at
# batch size 1). Report this caveat alongside any reproducibility claim.
LLM_TEMPERATURE = 0.0
LLM_SEED = int(os.getenv("LLM_SEED", "42"))
# Gemini's OpenAI-compatible endpoint rejects an unrecognized `seed` field
# with HTTP 400 (measured) rather than ignoring it, so it must not be sent to
# that provider at all. Call sites do `**LLM_SEED_KWARGS` instead of passing
# seed=LLM_SEED directly.
LLM_SEED_KWARGS = {} if LLM_PROVIDER == "gemini" else {"seed": LLM_SEED}

# Per-request HTTP timeout (seconds) for the OpenAI client. Without this, the
# SDK's default timeout is long enough that a stuck/overloaded backend (e.g.
# a self-hosted vLLM server returning HTTP 500 under load) can leave a single
# call hanging for minutes before the retry loop even gets to try again —
# measured: one round on qwen took 284.7s wall-clock for what should have
# been a ~20-30s call, because of exactly this. A slow-but-healthy call with
# a large completion (4429 tokens) was measured at ~80s, so the timeout is
# set well above that rather than near the common case.
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "120"))

# Completion token budget. qwen's context used to be a hard 8192-token TOTAL
# (input+completion) ceiling that safe_max_tokens() below had to shrink
# per-call for — much smaller than Gemini/DeepSeek, where LLM_MAX_TOKENS was
# always a plain completion budget with the provider managing its own (much
# larger) context window server-side. qwen's server-side context (vLLM
# --max-model-len) has since been raised to match, so it now gets the same
# plain-completion-budget treatment as every other provider — no more
# subtracting the prompt from a shared window.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "14096"))


def safe_max_tokens(messages, floor=64, ceiling=None):
    """
    Completion-token budget. A no-op for every provider now: each one's
    context is large enough (server-managed) that LLM_MAX_TOKENS alone is a
    safe completion budget, so this always returns ceiling or LLM_MAX_TOKENS.
    Kept as the single choke point so a future small-context provider (a
    shared input+completion window, as qwen used to be at 8192) can have
    per-call shrinking added back here without touching call sites.
    """
    return ceiling or LLM_MAX_TOKENS


N_REFINEMENT_ROUNDS = 5    # Số vòng lặp iterative refinement

# ============================================================
# Two research scenarios for the SAME attack, toggled by one flag
# ============================================================
# SHOW_NEIGHBOR_PROFILES controls whether the prompt includes a "Profile of
# anonymized placeholders" block — each neighbor MID's facts pre-grouped
# under one label (see _build_profile_block in step4_llm_inference.py).
#
#   True  (attacker-with-tooling scenario): mirrors an attacker who groups a
#     scraped graph's facts by entity before analysing it (every fact shown is
#     already present as raw evidence; grouping just removes manual busywork).
#     Makes recognising a neighbor from prior knowledge easier — legitimate
#     for this scenario, since that background knowledge is what it measures.
#   False (pure graph-structure scenario): the model sees only raw evidence
#     triples, un-grouped — isolates how much the anonymized graph's
#     STRUCTURE alone gives away.
#
# Neither setting is "more correct" — they measure different things. Run
# both and report both; don't average them into one number.
#
# Default False; set SHOW_NEIGHBOR_PROFILES=true in the environment to switch
# to the attacker-with-tooling scenario.
SHOW_NEIGHBOR_PROFILES = os.getenv("SHOW_NEIGHBOR_PROFILES", "false").lower() == "true"

# Agentic retrieval budget — how much NEW evidence the LLM can pull per round
# when it issues EXPLORE / NEED_RELATION requests.
MAX_NEW_TRIPLES_PER_ROUND = 25   # was hard-coded 15 in run_deanon_attack.py
MAX_EXPLORE_PER_ROUND = 5        # was hard-coded 2 (per-placeholder Step 2 cap)

# ============================================================
# Evidence Quality (Step 3 — TF-IDF style scoring)
# ============================================================
# Triple score = relation_idf[r] + alpha if connects to context entity
#                                 + beta  if r in selected_relations
# Triples are sorted by this score (descending) before truncation to
# MAX_EVIDENCE_TRIPLES.
# NOTE: inert while EVIDENCE_SELECTION == "hop" (that branch returns before
# scoring runs), but still imported by step3, so must stay defined.
EVIDENCE_ALPHA = 1.5         # Bonus for triples connecting to a sensitive context entity
EVIDENCE_BETA = 1.0          # Bonus for triples whose relation was selected in Step 2
EVIDENCE_GAMMA = 2.0         # Discrimination bonus: GAMMA / sharing_count for rare (r, value) pairs

# --- Connectivity-aware scoring (Phase D) ---------------------------------
# Scoring each triple in isolation let high-IDF edges survive truncation while
# the edges linking them back to the target were cut, leaving orphan
# fragments (a real name visible but structurally unreachable).
EVIDENCE_PROXIMITY = 2.5      # bonus / (1 + hops-from-target); nearer facts rank higher
EVIDENCE_MAX_HOP_KEEP = 4     # only award the proximity bonus within this radius
EVIDENCE_REQUIRE_CONNECTED = True  # after truncation, drop triples not reachable from target

# --- Evidence selection strategy (Phase D) --------------------------------
#   "scored"  : the hand-tuned ranking above (IDF + context/target/selected/rarity
#               bonuses + proximity). Encodes OUR assumptions about what matters.
#   "hop"     : no semantic scoring — keep facts strictly by distance from the
#               target (all 1-hop, then 2-hop, ...). Hands the model the raw
#               local neighbourhood and lets it decide what's informative,
#               which is what we want when measuring the model's own reasoning.
EVIDENCE_SELECTION = "hop"   # "scored" | "hop"

# Entities above this degree percentile are skipped in multi-hop, to keep
# real hubs (United States, USD, marriage, ...) from swamping the evidence.
HUB_PERCENTILE = 99.9

# Restrict multi-hop expansion to person-likely MID entities.
# TURNED OFF: measured that structural paths from a MID key to its real-name
# key almost always run through named string nodes (place/org names), not
# through a chain of MIDs — with this filter on, the real name was mostly
# unreachable. Off = noisier evidence, but the name can actually appear.
PERSON_ONLY_MULTIHOP = False

# --- Skip anonymized MIDs entirely during expansion ------------------------
# Resolving OTHER MIDs is mostly a dead end: a MID's own facts are exactly the
# generic attributes the prompt already forbids as grounding, so identifying
# one yields another un-nameable MID. Skipping them costs a small fraction of
# victims their clean path, but buys back a meaningful share of the evidence
# budget for facts carrying real names. The TARGET's own MID and its sensitive
# context are always exempt — those are the problem statement, not noise.
SKIP_MID_IN_EXPANSION = True

# --- Evidence rendering ----------------------------------------------------
#   "triple"    : raw ['head', 'relation', 'tail'] lines + one format note, as in
#                 KG-GPT. Keeps the FULL relation path, so the subject's type
#                 and edge direction survive.
#   "sentence"  : verbalizes each triple, shortening the relation to its last
#                 path component — readable, but collapses distinctions (e.g.
#                 ethnicity vs. a place-level statistic) that this attack
#                 turns on.
EVIDENCE_FORMAT = "triple"   # "triple" | "sentence"

# --- Drop place-to-place edges before retrieval ----------------------------
# Relations that can only ever join a PLACE to a PLACE (or a population
# statistic) can never name a person, so keeping them in evidence is budget
# spent on something that can't answer the question. Pruning removes only a
# small fraction of all triples and cost no victims their real-name edges in
# testing, while measurably raising GT-in-evidence rate at every budget size.
PRUNE_GEO_RELATIONS = True
GEO_RELATION_PREFIXES = (
    "/location/location/adjoin",
    "/location/location/contains",
    "/location/location/time_zones",
    "/location/statistical_region",
    "/location/administrative_division",
    "/location/country/second_level_divisions",
    "/location/capital_of_administrative_division",
    "/location/hud_",
    "/location/us_county",
    "/base/biblioness",
    "/base/aareas",
    "/user/tsegaran",
)
