"""UNFOLD — multi-anchor predicate-budget subgraph search.

Pipeline:
  EXTRACT   parse_question + ground_bag
            → bag of entities + predicates (with KG-grounded ids if available)
            (≤ 2 LLM calls: bag-extraction, predicate-grounding)

  CONSTRUCT search()
            → subgraph (set of triples) via multi-anchor predicate-budget BFS:
              - KG short-circuit per (entity, predicate) when grounded
              - Context-fill (batched per entity, parallel) on KG misses

  WALK      a set of rules that refine the constructed subgraph into a
            smaller, organized form for the next step

  POSE      LLM call within compose()
            → final answer string

State inside CONSTRUCT: (entity, consumed_predicates, source_anchor)
  - consumed_predicates: predicate indices this path has burned
  - source_anchor: anchor this path originated from
"""
from __future__ import annotations
import json, os, re, sqlite3, sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Tuple, Optional, FrozenSet, Any

# LLM sampling temperatures. Both default to 0.0 (greedy decoding)
# and can be independently overridden via env vars.
# - _LLM_TEMPERATURE: structured-output stages (extract/ground/filter/coref/
#   context-fill/leaf-pick). Lower keeps JSON stable.
# - _COMPOSE_TEMPERATURE: compose stage (sub-query yes/no + final answer).
_LLM_TEMPERATURE = float(os.environ.get("UNFOLD_TEMPERATURE", "0.0"))
_COMPOSE_TEMPERATURE = float(os.environ.get("UNFOLD_COMPOSE_TEMPERATURE",
                                              str(_LLM_TEMPERATURE)))

# Tolerant JSON parser: handles LLM output glitches (trailing commas, single
# quotes, unescaped chars). Falls back to stdlib json on failure.
try:
    import json5 as _json5
except ImportError:
    _json5 = None


def _safe_loads(text: str):
    """Parse JSON tolerantly. Try json5 first if available, then stdlib."""
    if _json5 is not None:
        try:
            return _json5.loads(text)
        except Exception:
            pass
    return json.loads(text)

# ============================================================
# KG access + per-paragraph triple extraction
# ============================================================

# --------------------------------------------------------------------------
# KG predicate vocabulary
# --------------------------------------------------------------------------
_KG_PREDICATE_CACHE: Optional[List[str]] = None
_KG_PREDICATE_SET_LOWER: Optional[Set[str]] = None


def _kg_predicate_list(conn) -> List[str]:
    """Distinct predicate names from the KG. Cached at process scope."""
    global _KG_PREDICATE_CACHE, _KG_PREDICATE_SET_LOWER
    if _KG_PREDICATE_CACHE is None:
        try:
            rows = conn.execute(
                "SELECT predicate, COUNT(*) c FROM triples "
                "GROUP BY predicate ORDER BY c DESC").fetchall()
            _KG_PREDICATE_CACHE = [p for (p, _c) in rows]
            _KG_PREDICATE_SET_LOWER = {p.lower() for p in _KG_PREDICATE_CACHE}
        except Exception:
            _KG_PREDICATE_CACHE = []
            _KG_PREDICATE_SET_LOWER = set()
    return _KG_PREDICATE_CACHE


def _is_kg_predicate(predicate: str) -> bool:
    """True if `predicate` matches a KG predicate verbatim (case-insensitive)."""
    if _KG_PREDICATE_SET_LOWER is None:
        return False
    return predicate.lower() in _KG_PREDICATE_SET_LOWER


# --------------------------------------------------------------------------
# KG connection + entity cache
# --------------------------------------------------------------------------
_KG_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "local_graph", "knowledge_graph.db")
_kg_conn = None
_ENTITIES_LOWER_TO_NAME: Dict[str, str] = {}
_ENTITIES_CACHE_BUILT = False


def _get_kg() -> Optional[sqlite3.Connection]:
    """Open a connection to the local KG, or None if disabled/missing.

    Set `UNFOLD_NO_KG=1` to skip KG access entirely (the pipeline then
    relies only on the question's context paragraphs).
    """
    global _kg_conn
    if os.environ.get("UNFOLD_NO_KG", "").lower() in ("1", "true", "yes"):
        return None
    if _kg_conn is not None:
        return _kg_conn
    if os.path.exists(_KG_DB_PATH):
        try:
            _kg_conn = sqlite3.connect(_KG_DB_PATH, check_same_thread=False)
            return _kg_conn
        except Exception:
            return None
    return None


def _build_entity_cache() -> None:
    """One-time load of all KG entity names into an in-memory dict."""
    global _ENTITIES_CACHE_BUILT
    if _ENTITIES_CACHE_BUILT:
        return
    kg = _get_kg()
    if kg is None:
        _ENTITIES_CACHE_BUILT = True
        return
    for (ent,) in kg.execute("SELECT name FROM entities"):
        _ENTITIES_LOWER_TO_NAME[ent.lower()] = ent
    _ENTITIES_CACHE_BUILT = True


def kg_resolve_entity(name: str) -> Optional[str]:
    """Resolve a free-text name to the KG's canonical entity name, or None.

    Two strategies, in order:
      1. exact (case-insensitive) match;
      2. strip a trailing parenthetical disambiguator (e.g.
         "Foo (politician)" -> "Foo") and accept any canonical whose token
         set is a superset of the stripped form's tokens with at most 3
         extra tokens (e.g. "George Cochrane" -> "George Augustus Frederick
         Cochrane").
    """
    kg = _get_kg()
    if kg is None:
        return None
    if not _ENTITIES_CACHE_BUILT:
        _build_entity_cache()
    name_lower = name.lower()
    hit = _ENTITIES_LOWER_TO_NAME.get(name_lower)
    if hit is not None:
        return hit
    stripped = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
    base = stripped.lower() if stripped else name_lower
    base_tokens = set(t for t in base.split() if t)
    if len(base_tokens) < 2:
        return None
    for ent_lower, ent in _ENTITIES_LOWER_TO_NAME.items():
        ent_tokens = set(ent_lower.split())
        if base_tokens.issubset(ent_tokens) and len(ent_tokens) - len(base_tokens) <= 3:
            return ent
    return None


def kg_pull_neighborhood(entity: str) -> List[Tuple[str, str, str]]:
    """Pull all triples in which `entity` is the subject or the object."""
    kg = _get_kg()
    if kg is None:
        return []
    triples = []
    for row in kg.execute(
        "SELECT subject, predicate, object FROM triples WHERE LOWER(subject) = LOWER(?)",
        (entity,)
    ).fetchall():
        triples.append((row[0], row[1], row[2]))
    for row in kg.execute(
        "SELECT subject, predicate, object FROM triples WHERE LOWER(object) = LOWER(?)",
        (entity,)
    ).fetchall():
        triples.append((row[0], row[1], row[2]))
    return triples


# --------------------------------------------------------------------------
# Context normalization
# --------------------------------------------------------------------------
def _normalize_context(context) -> List[Tuple[str, str]]:
    """Normalize a benchmark `context` field into `[(title, text), ...]`."""
    result = []
    for item in context:
        if isinstance(item, dict):
            title = item.get("title", "")
            text = item.get("paragraph_text", "")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            title = item[0]
            sents = item[1]
            text = " ".join(s.strip() for s in sents) if isinstance(sents, list) else str(sents)
        else:
            continue
        text = text.strip()
        if title and text:
            result.append((title, text))
    return result


# --------------------------------------------------------------------------
# Per-paragraph triple extraction
# --------------------------------------------------------------------------
_GUIDED_EXTRACT_PROMPT = (
    "Given the paragraph below, extract factual triplets (subject, predicate, value) "
    "that match the queried predicate.\n\n"
    "RULES:\n"
    "1. SUBJECT should be one of the listed Known entities, a clear alias of one "
    "(e.g. 'DiCaprio' for 'Leonardo DiCaprio', 'Wojna polsko-ruska' for "
    "'Polish-Russian War (Film)'), or another entity from the paragraph that is "
    "directly connected to a Known entity (so multi-hop chains can extend). "
    "Avoid extracting triples about entities only tangentially mentioned.\n"
    "2. VALUE must be a SPECIFIC named entity, exact date, or number. NEVER a vague "
    "phrase like 'early 1990s', 'late in the war', 'a movie', 'his role', 'a city'. "
    "If the text only gives a vague value, do NOT extract.\n"
    "3. Match the predicate by MEANING, not literal wording. Accept synonyms, "
    "paraphrases, and inverse phrasings (e.g. 'X is made from Y' for queried "
    "predicate 'used to make' → extract Y is used to make X). Always use the "
    "predicate label exactly as given in your output, even when the text used a "
    "different word.\n"
    "4. The triplet must be DIRECTLY supported by the text. Do NOT infer, guess, "
    "or apply background knowledge. If unsure, return [].\n"
    "5. Subject and value must be DIFFERENT entities.\n"
    "6. NEVER use 'None', 'null', empty strings, or vague placeholders for the "
    "subject, predicate, or value. If any field would be a placeholder, do not "
    "extract the triple.\n\n"
    "OUTPUT:\n"
    '- A JSON array of objects with keys "s", "p", "o".\n'
    "- Empty array [] if no clear, fully-supported triplet exists.\n"
    "- Return ONLY the JSON array. No markdown, no explanation."
)


def _guided_extract(title: str, text: str, predicate: str,
                    known_entities: List[str], client, model: str
                    ) -> List[Tuple[str, str, str]]:
    """Ask the LLM to extract triples matching `predicate` from one paragraph."""
    ent_list = ", ".join(f'"{e}"' for e in known_entities) if known_entities else "none"
    prompt = (
        f"{_GUIDED_EXTRACT_PROMPT}\n\n"
        f"Predicate: \"{predicate}\"\n"
        f"Known entities: [{ent_list}]\n\n"
        f"Paragraph title: {title}\nText: {text}"
    )
    return _call_extract(prompt, client, model)


def _call_extract(prompt: str, client, model: str) -> List[Tuple[str, str, str]]:
    """Shared LLM call for triple extraction."""
    try:
        response = client.chat.completions.create(
            model=model, temperature=_LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": "You extract structured facts from text."},
                {"role": "user", "content": prompt},
            ],
            extra_body={"reasoning": {"effort": "minimal"}},
        )
        usage = getattr(response, "usage", None)
        if usage:
            _token_counts["input"] += getattr(usage, "prompt_tokens", 0) or 0
            _token_counts["output"] += getattr(usage, "completion_tokens", 0) or 0
        _token_counts["calls"] += 1
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```[\w]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
        parsed = json.loads(raw)
        return [
            (str(item.get("s", "")).strip(), str(item.get("p", "")).strip(),
             str(item.get("o", "")).strip())
            for item in parsed
            if item.get("s") and item.get("p") and item.get("o")
            and str(item["s"]).strip().lower() != str(item["o"]).strip().lower()
        ]
    except Exception as e:
        print(f"      [extract ERROR] {e}")
        return []


# Type used by `unfold.py`'s walk: (entity, retired predicates, current
# predicate index, consecutive-repeat counter).
State = Tuple[str, FrozenSet[int], Optional[int], int]


# ============================================================
# EXTRACT: bag extraction + grounding
# ============================================================

_PARSE_V2_PROMPT = """You parse a natural-language question into two lists:

  "entities":   named real-world entities mentioned in the question (people,
                places, works, organizations, products). Keep full names
                intact, including titles and disambiguators. Include values
                used as constraints.

  "predicates": relations the question is about, each annotated with the
                TYPE of object the relation is expected to land on. Each
                predicate is an object with two fields:
                  "label": short canonical phrasing of the relation;
                           paraphrases are acceptable.
                  "type":  a soft, generic tag for the relation's object —
                           pick whichever best fits the kind of thing the
                           object should be. Use generic tags only (a
                           handful of broad categories such as person,
                           place, institution, year, work, number); do not
                           use specific instance names as types.
                List predicates in the natural order they chain to reach
                the answer. A predicate may itself be composed of multiple
                base relations — in that case, list each base relation
                separately.

Rules:
  - A predicate expresses a relation between two distinct things.
    Verbs and verbal phrases qualify. A noun qualifies ONLY if the
    relation inherently requires two participants — the noun makes
    no sense without both ends (one cannot exist in this relation
    without the other). A noun that describes a property, profession,
    category, or kind of ONE thing on its own is NOT a predicate —
    fold it into the `type` field of the predicate that retrieves it,
    or move it to `entities` if it names a specific value.
  - Use only POSITIVE base relations. Strip negations and ordering
    qualifiers from predicate labels.
  - Strip modifiers and comparison words from predicate labels: same,
    different, both, also, share, more, less, largest, etc.
  - Do NOT include the question word as an entity or predicate.

Output strict JSON with exactly these two keys.

Examples (placeholders A, B, C are stand-ins for real entities):

Q: "Where was the spouse of A born?"
{"entities":["A"],
 "predicates":[
   {"label":"spouse",         "type":"person"},
   {"label":"place of birth", "type":"place"}
 ]}

Q: "Are A and B in the same country?"
{"entities":["A","B"],
 "predicates":[{"label":"country","type":"place"}]}

Q: "Among A, B, and C, which has the largest population?"
{"entities":["A","B","C"],
 "predicates":[{"label":"population","type":"number"}]}

Q: "Did A NOT receive the award given to B?"
{"entities":["A","B"],
 "predicates":[{"label":"award received","type":"work"}]}

Q: "Who succeeded A's predecessor?"
{"entities":["A"],
 "predicates":[
   {"label":"predecessor","type":"person"},
   {"label":"successor",  "type":"person"}
 ]}

Output ONLY the JSON.
"""


@dataclass
class BagItem:
    label: str
    # For entities: kg_id is the canonical KG name (or None if not in KG).
    # For predicates: kg_id is unused; use kg_ids instead — a list of KG
    # predicates (direct OR inverse) that match this question predicate.
    kg_id: Optional[str] = None
    kg_ids: List[str] = field(default_factory=list)
    # For predicates: a soft generic type tag (e.g. "person", "place",
    # "institution", "year", "work", "number") describing what kind of
    # object the relation should land on. Used downstream in BFS context-
    # fill to score / filter candidate edges by object-type compatibility.
    # Empty string when unset (e.g. legacy parser output) — treated as
    # "no type filter" by the consumer.
    type: str = ""

    @property
    def grounded(self) -> bool:
        # Entity is grounded if kg_id is set; predicate is grounded if kg_ids non-empty.
        return self.kg_id is not None or bool(self.kg_ids)


@dataclass
class Bag:
    entities: List[BagItem] = field(default_factory=list)
    predicates: List[BagItem] = field(default_factory=list)

    @property
    def all_grounded(self) -> bool:
        return (all(e.grounded for e in self.entities)
                and all(p.grounded for p in self.predicates))


def parse_question(question: str, client, model: str,
                       context: Optional[List] = None) -> Optional[Bag]:
    """Extract (entities, predicates) from the question. Context is NOT used —
    entities/predicates are read from the question text alone, which is far
    faster (no large prompt) and just as accurate for benchmark questions
    where named entities appear literally in the question.
    Single LLM call. Returns None on parse failure."""
    user_msg = f"Question: {question}"
    parsed = None
    raw = ""
    for attempt in range(2):
        try:
            kwargs = {
                "model": model, "temperature": _LLM_TEMPERATURE,
                "messages": [
                    {"role": "system", "content": _PARSE_V2_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            }
            # Try JSON mode on first attempt; fall back to plain on second.
            if attempt == 0:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            raw = (resp.choices[0].message.content or "").strip()
            raw = re.sub(r"^```[\w]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw).strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                raw = m.group(0)
            parsed = _safe_loads(raw)
            break
        except Exception as e:
            if attempt == 0:
                continue  # retry without json_object mode
            print(f"[PARSE-V2 ERROR] {e}")
            return None
    if parsed is None:
        return None

    bag = Bag()
    for e in parsed.get("entities", []):
        label = e if isinstance(e, str) else (e.get("label") if isinstance(e, dict) else None)
        if label:
            bag.entities.append(BagItem(label=str(label)))
    for p in parsed.get("predicates", []):
        # Predicate can be a plain string (legacy) or {"label": ..., "type": ...}.
        if isinstance(p, str):
            label, ptype = p, ""
        elif isinstance(p, dict):
            label = p.get("label")
            ptype = p.get("type") or ""
        else:
            label, ptype = None, ""
        if label:
            bag.predicates.append(BagItem(label=str(label), type=str(ptype).strip().lower()))
    if not bag.entities or not bag.predicates:
        return None
    return bag


_GROUND_SYSTEM_PROMPT = (
    "You map question predicates to a vocabulary of KG predicates. For each "
    "question predicate, return ALL KG predicates from the provided list that "
    "express the same relation — either:\n"
    "  - DIRECTLY: same relation (paraphrase / synonym / inflection ok). "
    "    Example: q-pred 'works at' matches KG 'employer'.\n"
    "  - AS THE INVERSE: the KG predicate is the inverse direction, so the "
    "    same fact is expressed with subject and object swapped. "
    "    Example: q-pred 'parent of' matches KG 'child of' (inverse).\n"
    "\n"
    "Return all matching KG predicates per question predicate (empty list if "
    "none match). Output strict JSON: "
    '{"q_pred_1": ["kg_pred_a", "kg_pred_b"], "q_pred_2": [], ...}. '
    "Pick from the provided list only."
)


_PREDICATE_ALIASES_CACHE: Dict[str, Dict[str, List[str]]] = {}

def _load_predicate_aliases(src: Optional[str]) -> Dict[str, List[str]]:
    """Return {kg_pred: [q_pred aliases]} for a benchmark, or {} if no map.
    Cached. Currently only Drowzee has a curated map."""
    if not src:
        return {}
    key = src.split('_')[0] if src.startswith('drowzee') else src
    if key in _PREDICATE_ALIASES_CACHE:
        return _PREDICATE_ALIASES_CACHE[key]
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, f"{key}_data", "predicate_aliases.json")
    out: Dict[str, List[str]] = {}
    try:
        if os.path.exists(path):
            data = json.load(open(path))
            out = data.get('kgpred_to_qpreds', {}) or {}
    except Exception:
        out = {}
    _PREDICATE_ALIASES_CACHE[key] = out
    return out


def _entity_kg_predicates(entity_name: str, q_pred_labels: List[str],
                            hard_cap: int = 30) -> List[str]:
    """Return distinct KG predicates this entity has, both directions. If the
    distinct count exceeds hard_cap, rank by token-overlap with the question's
    predicate labels and keep the top hard_cap. (No frequency-based ranking —
    high-frequency predicates tend to be metadata, not the relations users
    ask about.)"""
    triples = _kg_neighbors(entity_name)
    if not triples:
        return []
    distinct = list({str(t[1]) for t in triples if len(t) >= 3})
    if len(distinct) <= hard_cap:
        return distinct
    # Rank by token-overlap with question predicates only when we exceed cap.
    q_toks: Set[str] = set()
    for p in q_pred_labels:
        q_toks |= set(re.findall(r"\w+", p.lower()))
    def _score(p: str) -> int:
        return len(set(re.findall(r"\w+", p.lower())) & q_toks)
    distinct.sort(key=lambda p: (-_score(p), p))
    return distinct[:hard_cap]


def ground_bag(bag: Bag, question: str, client, model: str,
               hard_cap_per_entity: int = 20,
               common_top_n: int = 30,
               src: Optional[str] = None) -> Bag:
    """Attach KG IDs to entities. For predicates, build a hybrid shortlist:
      (a) per-entity distinct KG predicates (top-K per anchor)
      (b) the K' most-frequent KG predicates (covers intermediates / common
          relations like place_of_birth that don't appear in the anchor's
          neighborhood when the anchor is, say, a film)
    Union and dedupe. Let the LLM map question predicates to that focused
    list (direct or inverse). If NO entity is grounded, skip predicate
    grounding entirely — CONSTRUCT will rely on context-fill."""
    for e in bag.entities:
        e.kg_id = kg_resolve_entity(e.label)

    grounded_entities = [e for e in bag.entities if e.grounded]
    if not grounded_entities:
        return bag  # nothing to ground predicates against

    # (a) Per-entity distinct predicates.
    q_pred_labels = [p.label for p in bag.predicates]
    shortlist_set: Set[str] = set()
    for e in grounded_entities:
        for p in _entity_kg_predicates(e.label, q_pred_labels,
                                          hard_cap=hard_cap_per_entity):
            shortlist_set.add(p)

    # (b) Top-N most-frequent KG predicates (handles intermediates whose
    #     predicates don't appear in any anchor's neighborhood — e.g. when
    #     the anchor is a film, "place of birth" needed for chain-extension
    #     to a person doesn't appear via the film, but is a common KG pred).
    conn = _get_kg()
    if conn is not None:
        for p in _kg_predicate_list(conn)[:common_top_n]:
            shortlist_set.add(p)

    shortlist = sorted(shortlist_set)
    if not shortlist:
        return bag

    aliases_map = _load_predicate_aliases(src)
    def _annotate(kg_pred: str) -> str:
        als = aliases_map.get(kg_pred) or []
        return f"{kg_pred} ({', '.join(als)})" if als else kg_pred
    annotated_shortlist = [_annotate(p) for p in shortlist]

    pred_labels = [p.label for p in bag.predicates]
    user_msg = (
        f"Question: {question}\n"
        f"Question predicates: {json.dumps(pred_labels)}\n"
        f"KG predicates (pick from these): {json.dumps(annotated_shortlist)}"
    )
    parsed = None
    for attempt in range(2):
        try:
            kwargs = {
                "model": model, "temperature": _LLM_TEMPERATURE,
                "messages": [
                    {"role": "system", "content": _GROUND_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            }
            if attempt == 0:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            raw = (resp.choices[0].message.content or "").strip()
            raw = re.sub(r"^```[\w]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw).strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                raw = m.group(0)
            parsed = _safe_loads(raw)
            break
        except Exception:
            if attempt == 0:
                continue
            return bag  # grounding failed → leave predicates ungrounded

    if parsed is None:
        return bag

    valid = set(shortlist)
    valid_lower = {s.lower(): s for s in shortlist}
    # Reverse alias index: any alias or annotated form maps back to canonical KG label.
    alias_to_canonical: Dict[str, str] = {}
    for kg_pred in shortlist:
        for al in aliases_map.get(kg_pred, []) or []:
            alias_to_canonical[al.lower()] = kg_pred

    def _canonicalize(x: str) -> Optional[str]:
        if not isinstance(x, str): return None
        s = x.strip()
        if not s or s.upper() == "NONE": return None
        # Strip annotated-form suffix: "subclass of (is part of)" → "subclass of"
        s_strip = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
        if s_strip in valid: return s_strip
        if s_strip.lower() in valid_lower: return valid_lower[s_strip.lower()]
        if s_strip.lower() in alias_to_canonical: return alias_to_canonical[s_strip.lower()]
        return None

    for p in bag.predicates:
        v = parsed.get(p.label)
        canon: List[str] = []
        if isinstance(v, list):
            for x in v:
                c = _canonicalize(x)
                if c and c not in canon: canon.append(c)
        elif isinstance(v, str):
            c = _canonicalize(v)
            if c: canon.append(c)
        p.kg_ids = canon
        # Defensive: if the LLM dropped a literal exact-label match that's
        # right there in the shortlist, force-include it. Catches cases where
        # the grounding LLM returns [] for a q-pred whose canonical label is
        # already in the KG vocabulary.
        canonical = valid_lower.get(p.label.lower())
        if canonical and canonical not in p.kg_ids:
            p.kg_ids.append(canonical)
    return bag


# ============================================================
# RELEVANCE FILTER: one LLM call per question reduces the context to the
# top-K paragraphs most likely to contain the answer. Replaces the
# literal-mention substring filter at the global level — that filter still
# runs per-anchor inside `_batched_context_fill`, but on a much smaller
# set, so it becomes a near-no-op. Catches acronyms / aliases / anaphora
# the substring filter misses (e.g. "FERC" vs "Federal Energy Regulation
# Commission").
# ============================================================

_FILTER_PARAGRAPHS_PROMPT = """You filter paragraphs for relevance to a question. Output the IDs of the paragraphs most likely to contain the information needed to answer the question.

OUTPUT: strict JSON of shape {"ids": [<int>, ...]}. List paragraph IDs in decreasing order of relevance, at most {top_k}. If fewer paragraphs are relevant, list only those.
"""


def _llm_filter_paragraphs(question: str, context,
                              client, model: str, top_k: int = 5,
                              verbose: bool = False
                              ) -> List[Tuple[str, str]]:
    """Reduce a list of paragraphs to the top_k most relevant for the question
    via one LLM call. Returns (title, text) tuples in the same shape as
    `_normalize_context`. Falls back to the first top_k paragraphs on any
    LLM/parse failure."""
    paragraphs = _normalize_context(context) if context else []
    if not paragraphs or len(paragraphs) <= top_k:
        return paragraphs

    ctx_blob = "\n\n".join(
        f"[{i+1}] {title}\n{text}" for i, (title, text) in enumerate(paragraphs)
    )
    user_msg = (
        f"Question: {question}\n\n"
        f"=== PARAGRAPHS ===\n{ctx_blob}\n\n"
        f'Return JSON of shape: {{"ids": [<int>, ...]}}'
    )
    system_prompt = _FILTER_PARAGRAPHS_PROMPT.replace("{top_k}", str(top_k))

    parsed = None
    for attempt in range(2):
        try:
            kwargs = {
                "model": model, "temperature": _LLM_TEMPERATURE,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
            }
            if attempt == 0:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            raw = (resp.choices[0].message.content or "").strip()
            raw = re.sub(r"^```[\w]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw).strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                parsed = _safe_loads(m.group(0))
            else:
                raise ValueError("no JSON object")
            break
        except Exception as e:
            if attempt == 0:
                continue
            if verbose:
                print(f"  [FILTER ERROR] {e} — falling back to first {top_k}")
            return paragraphs[:top_k]

    if not isinstance(parsed, dict):
        return paragraphs[:top_k]
    raw_ids = parsed.get("ids", [])
    if not isinstance(raw_ids, list):
        return paragraphs[:top_k]

    kept: List[Tuple[str, str]] = []
    seen: Set[int] = set()
    for x in raw_ids:
        try:
            idx = int(x) - 1   # IDs are 1-based in the prompt
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(paragraphs) and idx not in seen:
            kept.append(paragraphs[idx])
            seen.add(idx)
        if len(kept) >= top_k:
            break

    if not kept:
        return paragraphs[:top_k]

    if verbose:
        print(f"  [filter] {len(paragraphs)} paragraphs → kept top {len(kept)} "
              f"(IDs: {sorted(i+1 for i in seen)})")
    return kept


# ============================================================
# COREFERENCE ANNOTATION: anchor-aware inline-bracketing pre-pass
#
# Runs once per question between EXTRACT and CONSTRUCT. The LLM is asked to
# emit a COMPACT list of (paragraph_id, span_text, anchor_label) annotations,
# NOT full paragraph rewrites — this cuts output tokens dramatically. Python
# then applies each annotation by inserting " [anchor]" after the first
# occurrence of the span in the paragraph.
#
# A cheap pre-filter skips paragraphs that have neither an anchor reference
# nor any anaphora signal (pronouns, acronyms, role titles) — pure distractor
# paragraphs cost no LLM tokens.
#
# The existing literal-mention paragraph filter (`_filter_paragraphs_for_entity`)
# then picks up the inserted bracketed names automatically — no ranker change.
# ============================================================

_RESOLVE_COREF_PROMPT = """You identify references in paragraphs that point to a list of FOCUS entities.

For each paragraph, find spans (pronouns, acronyms, role titles, partial names, definite descriptions) that coreferentially refer to one of the FOCUS entities. Emit them as a list of annotations — DO NOT rewrite the paragraphs.

Rules:
- Only emit an annotation when the coreference is unambiguous from the paragraph itself. Do not infer from background knowledge.
- The "span" must be a literal substring of the paragraph text.
- The "anchor" must be one of the supplied focus entities, written exactly as in the focus list.
- If a focus entity's full name already appears literally in the paragraph, do not emit an annotation for that literal mention.
- Spans that do not refer to any focus entity must be skipped.

OUTPUT: strict JSON of shape {"annotations": [{"id": <paragraph_id>, "span": "<exact substring>", "anchor": "<focus entity name>"}, ...]}. Empty list if no coreferences are found.
"""


# Pronouns + role-titles: case-insensitive ("he", "He", "HE" all match).
_ANAPHORA_PRONOUN_PATTERN = re.compile(
    r"\b(he|she|it|they|him|her|them|his|hers|their|theirs|its)\b"
    r"|\bthe\s+(?:VP|CEO|chairman|chairwoman|director|founder|firm|group|"
    r"board|judge|officer|analyst|writer|reporter|speaker|company|agency|"
    r"commission)\b",
    flags=re.IGNORECASE,
)
# Acronyms: case-SENSITIVE — must be 2-6 uppercase letters. Without this flag
# `[A-Z]{2,6}` with IGNORECASE matches any 2-6 letter word, defeating the filter.
_ANAPHORA_ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,6}\b")


def _should_coref_paragraph(text: str, anchor_lowers: List[str]) -> bool:
    """Quick check: does this paragraph have anything coref could resolve?

    Keep the paragraph if EITHER (a) at least one anchor appears literally
    (the paragraph is likely on-topic and may contain anaphora pointing into
    or out of the anchor), OR (b) the paragraph contains anaphora signals
    (pronouns / role-titles / acronyms) that might refer to an anchor.
    Skip paragraphs with neither — they're pure distractors."""
    if not text:
        return False
    text_l = text.lower()
    if any(a in text_l for a in anchor_lowers):
        return True
    if _ANAPHORA_PRONOUN_PATTERN.search(text):
        return True
    if _ANAPHORA_ACRONYM_PATTERN.search(text):
        return True
    return False


_TRIVIAL_ARTICLE_RE = re.compile(r"^(the|a|an)\s+", flags=re.IGNORECASE)


def _is_trivial_annotation(span: str, anchor: str) -> bool:
    """Drop annotations the LLM shouldn't have emitted: span equals anchor,
    or span equals anchor with a leading definite/indefinite article.
    The prompt forbids these, but the LLM sometimes emits them anyway."""
    s = span.strip().lower()
    a = anchor.strip().lower()
    if not s or not a:
        return True
    if s == a:
        return True
    if _TRIVIAL_ARTICLE_RE.sub("", s) == a:
        return True
    if _TRIVIAL_ARTICLE_RE.sub("", a) == s:
        return True
    return False


def _apply_annotations(text: str, anns: List[Dict[str, str]],
                         anchor_set: Set[str]) -> str:
    """Apply LLM-emitted annotations to a paragraph by inserting ' [anchor]'
    after the first case-insensitive occurrence of each span. Drops:
      - annotations whose anchor isn't in `anchor_set` (LLM format guard)
      - trivial self-annotations where span IS the anchor's literal name
        (LLM ignoring an explicit prompt rule; adds no searchable signal)"""
    out = text
    for a in anns:
        span = a.get("span", "")
        anchor = a.get("anchor", "")
        if not span or not anchor:
            continue
        if anchor not in anchor_set:
            continue
        if _is_trivial_annotation(span, anchor):
            continue
        idx = out.lower().find(span.lower())
        if idx < 0:
            continue
        end = idx + len(span)
        out = out[:end] + f" [{anchor}]" + out[end:]
    return out


def _resolve_coreferences(context, anchor_labels: List[str],
                            client, model: str,
                            verbose: bool = False
                            ) -> List[Tuple[str, str]]:
    """Compact-output coreference annotation. The LLM emits a list of
    (paragraph_id, span, anchor) annotations; Python inserts ' [anchor]'
    after each span in the original paragraph text. Paragraphs without any
    anchor mention or anaphora signal are pre-filtered out and pass through
    unchanged. Returns (title, annotated_text) tuples in input order. Falls
    back to the original normalized context on parsing / LLM failure."""
    paragraphs = _normalize_context(context) if context else []
    if not paragraphs or not anchor_labels:
        return paragraphs

    anchor_set = set(anchor_labels)
    anchor_lowers = [a.lower() for a in anchor_labels]

    # Pre-filter: only paragraphs with at least one anchor or anaphora signal
    keep_indices = [i for i, (_t, txt) in enumerate(paragraphs)
                    if _should_coref_paragraph(txt, anchor_lowers)]
    if not keep_indices:
        if verbose:
            print(f"  [coref] no in-scope paragraphs ({len(paragraphs)} skipped)")
        return paragraphs

    # Build prompt using ORIGINAL indices (1-based) so the LLM can refer back.
    ctx_blob = "\n\n".join(
        f"[{i+1}] {paragraphs[i][0]}\n{paragraphs[i][1]}" for i in keep_indices
    )
    user_msg = (
        f"Focus entities: {json.dumps(anchor_labels)}\n\n"
        f"=== PARAGRAPHS ===\n{ctx_blob}\n\n"
        f"Return JSON of shape: "
        f'{{"annotations": [{{"id": <int>, "span": "<substring>", '
        f'"anchor": "<focus entity>"}}, ...]}}'
    )
    parsed = None
    for attempt in range(2):
        try:
            kwargs = {
                "model": model, "temperature": _LLM_TEMPERATURE,
                "messages": [
                    {"role": "system", "content": _RESOLVE_COREF_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            }
            if attempt == 0:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            raw = (resp.choices[0].message.content or "").strip()
            raw = re.sub(r"^```[\w]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw).strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                parsed = _safe_loads(m.group(0))
            else:
                raise ValueError("no JSON object found")
            break
        except Exception as e:
            if attempt == 0:
                continue
            if verbose:
                print(f"  [COREF ERROR] {e} — falling back to original context")
            return paragraphs

    if parsed is None:
        return paragraphs

    anns_all = parsed.get("annotations", []) if isinstance(parsed, dict) else []
    # Group annotations by paragraph id
    anns_by_id: Dict[int, List[Dict[str, str]]] = {}
    for a in anns_all:
        if not isinstance(a, dict):
            continue
        try:
            pid = int(a.get("id"))
        except (TypeError, ValueError):
            continue
        anns_by_id.setdefault(pid, []).append(a)

    out: List[Tuple[str, str]] = []
    n_annotated = 0
    n_inserted = 0
    for i, (title, text) in enumerate(paragraphs):
        anns = anns_by_id.get(i + 1)
        if not anns:
            out.append((title, text))
            continue
        annotated = _apply_annotations(text, anns, anchor_set)
        if annotated != text:
            n_annotated += 1
            n_inserted += annotated.count("[") - text.count("[")
        out.append((title, annotated))

    if verbose:
        print(f"  [coref] sent {len(keep_indices)}/{len(paragraphs)} paragraphs, "
              f"{len(anns_all)} annotations returned, "
              f"{n_inserted} brackets inserted across {n_annotated} paragraphs")
    return out


# ============================================================
# CONSTRUCT: multi-anchor predicate-budget search → subgraph of triples
# ============================================================

# State during search:
#   (entity_name, consumed_pred_idxs: frozenset[int], anchor_origin: str)
State = Tuple[str, FrozenSet[int], str]


_KG_MAX_NEIGHBORS = int(os.environ.get("UNFOLD_V2_KG_CAP", "200"))


def _kg_neighbors(entity_name: str) -> List[Tuple[str, str, str]]:
    """KG neighborhood of an entity (both directions). Returns list of triples,
    capped at _KG_MAX_NEIGHBORS to prevent runaway expansion on dense entities."""
    resolved = kg_resolve_entity(entity_name)
    if not resolved:
        return []
    triples = kg_pull_neighborhood(resolved)
    if len(triples) > _KG_MAX_NEIGHBORS:
        triples = triples[:_KG_MAX_NEIGHBORS]
    return triples


def _kg_neighbors_for_predicates(entity_name: str,
                                    predicate_targets: List[str]
                                    ) -> List[Tuple[str, str, str]]:
    """Pull KG triples involving entity, restricted to the given predicate set
    at SQL level. Cap applied AFTER the predicate filter, so relevant edges
    aren't lost on dense anchors but pathologically broad intermediates
    (e.g. 'product' with thousands of subclasses) don't blow up the BFS."""
    if not predicate_targets:
        return []
    resolved = kg_resolve_entity(entity_name)
    if not resolved:
        return []
    kg = _get_kg()
    if kg is None:
        return []
    targets_lower = sorted({p.lower() for p in predicate_targets})
    placeholders = ",".join("?" * len(targets_lower))
    triples: List[Tuple[str, str, str]] = []
    for row in kg.execute(
        f"SELECT subject, predicate, object FROM triples "
        f"WHERE LOWER(subject) = LOWER(?) AND LOWER(predicate) IN ({placeholders})",
        (entity_name, *targets_lower)
    ).fetchall():
        triples.append((row[0], row[1], row[2]))
    for row in kg.execute(
        f"SELECT subject, predicate, object FROM triples "
        f"WHERE LOWER(object) = LOWER(?) AND LOWER(predicate) IN ({placeholders})",
        (entity_name, *targets_lower)
    ).fetchall():
        triples.append((row[0], row[1], row[2]))
    if len(triples) > _KG_MAX_NEIGHBORS:
        triples = triples[:_KG_MAX_NEIGHBORS]
    return triples


_BATCH_CONTEXT_FILL_PROMPT = """You extract factual triples (subject, predicate, object) from a CONTEXT for a specific entity.

CORE RULE — fact support: every emitted triple must express a FACT directly stated in the context. The subject and object must appear (literally or via clear coreference), and the relation must be the one the context describes, in the same direction. Forbidden: inventing facts the context doesn't state, inference across multiple sentences, transitive combination of two separate facts, background knowledge.

Predicate matching is SEMANTIC, not literal. When searching the context for a requested relation, match by MEANING — accept paraphrases, idiomatic phrasings, indirect expressions, and synonymous verbs that describe the same fact. The context may not use the same wording as the requested predicate label.

Emit each triple using the REQUESTED predicate label exactly as it appears in the input list. Do not substitute the context's wording. This makes the triple matchable against the question's predicate downstream.

Direction follows the context. If the context says "X did Y to Z", emit (X, requested_label, Z), not the reverse. Do not emit both directions of an asymmetric relation.

Coverage. Either subject or object must be the requested entity. Aliases, acronyms, and pronouns are acceptable when the context disambiguates them.

Each predicate is annotated with a generic type tag — a hint about what kind of object the relation should land on. Use the tag to prefer candidates of the right type. Do not treat it as a hard veto: if a triple clearly states the predicate's relation, emit it even when the object's type only loosely matches the tag.

Output strict JSON: {"triples": [[s, p, o], ...]}. Empty list if nothing qualifies.
"""


def _filter_paragraphs_for_entity(paragraphs: List[Tuple[str, str]],
                                    entity: str,
                                    max_paragraphs: int = 5) -> List[Tuple[str, str]]:
    """Keep paragraphs relevant to the entity. Score each by:
       (a) entity appears in the title (strongest signal), AND/OR
       (b) entity appears in the body text (count = strength)
    Sort by score, return top-N. Falls back to top-N raw if nothing matches."""
    e = entity.lower().strip()
    if not e:
        return paragraphs[:max_paragraphs]
    scored = []
    for title, text in paragraphs:
        title_l = (title or "").lower()
        body_l = (text or "").lower()
        # Score: title-match worth +10 (very strong); body mentions worth +1 each
        s = (10 if e in title_l else 0) + body_l.count(e)
        if s > 0:
            scored.append((s, title, text))
    if not scored:
        return paragraphs[:max_paragraphs]
    scored.sort(key=lambda x: -x[0])
    return [(t, x) for _, t, x in scored[:max_paragraphs]]


def _batched_context_fill(entity: str, predicates,
                            paragraphs: List[Tuple[str, str]],
                            client, model: str) -> List[Tuple[str, str, str]]:
    """Single LLM call: extract triples (s, p, o) where s == entity or o == entity,
    and p is in the requested predicate list, supported by paragraphs.

    `predicates` is a list of either:
      - plain strings (legacy: no type annotation), or
      - {"label": <str>, "type": <str>} dicts annotating the expected
        object type per predicate.

    Paragraphs are pre-filtered to those that mention the entity, capped at 5,
    so the prompt stays short (~5x speedup on context-heavy questions)."""
    if not paragraphs or not predicates:
        return []
    paragraphs = _filter_paragraphs_for_entity(paragraphs, entity, max_paragraphs=5)
    ctx_blob = "\n\n".join(
        f"[{i+1}] {title}\n{text}"
        for i, (title, text) in enumerate(paragraphs)
    )
    # Normalize predicates → list of {label, type} dicts for the prompt.
    norm_preds = []
    for p in predicates:
        if isinstance(p, dict) and p.get("label"):
            norm_preds.append({"label": str(p["label"]),
                               "type": str(p.get("type") or "")})
        elif isinstance(p, str) and p.strip():
            norm_preds.append({"label": p, "type": ""})
    user_msg = (
        f"Entity: {entity}\n"
        f"Predicates: {json.dumps(norm_preds)}\n\n"
        f"=== CONTEXT ===\n{ctx_blob}"
    )
    # Wrap a list payload in an object for json_object mode compatibility.
    user_msg_obj = user_msg + (
        "\n\nReturn JSON of shape: "
        '{"triples": [[s, p, o], [s, p, o], ...]}.'
    )
    triples = None
    for attempt in range(2):
        try:
            kwargs = {
                "model": model, "temperature": _LLM_TEMPERATURE,
                "messages": [
                    {"role": "system", "content": _BATCH_CONTEXT_FILL_PROMPT},
                    {"role": "user", "content": user_msg_obj},
                ],
            }
            if attempt == 0:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            raw = (resp.choices[0].message.content or "").strip()
            raw = re.sub(r"^```[\w]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw).strip()
            # Try to find {"triples": [...]} or just [...]
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                obj = _safe_loads(m.group(0))
                triples = obj.get("triples", [])
            else:
                m2 = re.search(r"\[.*\]", raw, re.DOTALL)
                if m2:
                    triples = _safe_loads(m2.group(0))
                else:
                    raise ValueError("no JSON object or array found")
            break
        except Exception as e:
            if attempt == 0:
                continue
            print(f"  [BATCH-FILL ERROR] {e}")
            return []
    if triples is None:
        return []
    # Cap output at 5 triples per entity (across all predicates) to bound the
    # cascade at the next BFS hop. Take the first 5 distinct triples that
    # pass the post-LLM sanity filter.
    out = []
    seen: Set[Tuple[str, str, str]] = set()
    MAX_TRIPLES_PER_ENTITY = 5
    ent_norm = _norm_for_match(entity)
    for t in triples:
        if not (isinstance(t, list) and len(t) >= 3):
            continue
        s_raw, p_raw, o_raw = str(t[0]), str(t[1]), str(t[2])
        s_norm = _norm_for_match(s_raw)
        o_norm = _norm_for_match(o_raw)
        p_norm = _norm_for_match(p_raw)
        # Reject empty parts.
        if not s_norm or not o_norm or not p_norm:
            continue
        # Reject self-loops (subject == object after normalization).
        if s_norm == o_norm:
            continue
        # Reject triples where both endpoints are the requested entity
        # (LLM treating the entity as both sides of a relation about itself).
        if s_norm == ent_norm and o_norm == ent_norm:
            continue
        # Reject triples that don't actually involve the requested entity
        # at either endpoint (rule 2 enforcement at the code level).
        if ent_norm and s_norm != ent_norm and o_norm != ent_norm:
            # Allow if a substring containment is plausible (handles slight
            # alias differences the LLM might have produced).
            if ent_norm not in s_norm and ent_norm not in o_norm \
               and s_norm not in ent_norm and o_norm not in ent_norm:
                continue
        key = (s_raw, p_raw, o_raw)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
        if len(out) >= MAX_TRIPLES_PER_ENTITY:
            break
    return out


def _pred_matches(triple_pred: str, target_pred: str) -> bool:
    """Strict-ish predicate match: exact or word-set equality, not loose substring."""
    a = str(triple_pred).lower().strip()
    b = str(target_pred).lower().strip()
    if a == b:
        return True
    # word-set equality (handles "place of birth" vs "birth place")
    aw = set(re.findall(r'\w+', a))
    bw = set(re.findall(r'\w+', b))
    if aw == bw and aw:
        return True
    # subset only if no extra words on either side
    if aw and bw and (aw <= bw or bw <= aw):
        # accept only if difference is small (avoid "area" matching
        # "MusicBrainz area ID" which differs by 3 words)
        if abs(len(aw) - len(bw)) <= 1:
            return True
    return False




def _other_endpoint(triple: Tuple[str, str, str], ent: str) -> str:
    """Given a triple and a known endpoint, return the other endpoint."""
    s, p, o = triple
    if s.lower().strip() == ent.lower().strip():
        return o
    return s


_LINK_UNCOVERED_PROMPT = """You connect uncovered entities to an existing knowledge subgraph.

You are given:
  (a) GRAPH: a list of triples (subject, predicate, object) already collected for a question.
  (b) UNCOVERED: a list of entities from the question that do not appear in GRAPH.

Propose triples that link each UNCOVERED entity to an entity already in GRAPH (or to another UNCOVERED entity), using factual relations. Each triple must express a real fact and must use an entity that is either in GRAPH or in UNCOVERED on at least one endpoint.

Output strict JSON: {"triples": [[s, p, o], ...]}. Empty list if no plausible link exists."""


def _link_uncovered_entities(
        anchor_labels: List[str],
        edges: Set[Tuple[str, str, str]],
        client, model: str,
        verbose: bool = False) -> List[Tuple[str, str, str]]:
    """Optimization (paper §3, opt. 4): if some anchors do not appear in any
    edge of the collected subgraph, ask the LLM to propose triples that link
    them to the rest of the graph. Returns the list of validated new triples."""
    if not anchor_labels:
        return []
    in_graph = set()
    for s, _, o in edges:
        in_graph.add(_norm_for_match(s))
        in_graph.add(_norm_for_match(o))
    uncovered = [a for a in anchor_labels if _norm_for_match(a) not in in_graph]
    if not uncovered:
        return []
    graph_blob = "\n".join(f"({s}, {p}, {o})" for s, p, o in sorted(edges))
    user_msg = (
        f"UNCOVERED: {json.dumps(uncovered)}\n\n"
        f"=== GRAPH ===\n{graph_blob or '(empty)'}"
        "\n\nReturn JSON: {\"triples\": [[s, p, o], ...]}"
    )
    try:
        resp = client.chat.completions.create(
            model=model, temperature=_LLM_TEMPERATURE,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _LINK_UNCOVERED_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return []
        obj = _safe_loads(m.group(0)) or {}
        triples = obj.get("triples", []) or []
    except Exception as e:
        if verbose:
            print(f"  [LINK-UNCOVERED ERROR] {e}")
        return []
    out: List[Tuple[str, str, str]] = []
    uncovered_norms = {_norm_for_match(u) for u in uncovered}
    for t in triples:
        if not (isinstance(t, list) and len(t) >= 3):
            continue
        s_raw, p_raw, o_raw = str(t[0]), str(t[1]), str(t[2])
        if not s_raw.strip() or not p_raw.strip() or not o_raw.strip():
            continue
        if _norm_for_match(s_raw) == _norm_for_match(o_raw):
            continue
        endpoints = {_norm_for_match(s_raw), _norm_for_match(o_raw)}
        if not (endpoints & uncovered_norms):
            continue
        out.append((s_raw, p_raw, o_raw))
    if verbose and out:
        print(f"  [link-uncovered] {len(uncovered)} uncovered anchors, "
              f"added {len(out)} triples")
    return out


def search(bag: Bag, context, client, model: str,
           verbose: bool = False, skip_kg: bool = False,
           src: Optional[str] = None
           ) -> List[Tuple[str, str, str]]:
    """Multi-anchor predicate-budget BFS. Returns list of edge triples.
    skip_kg: when True, skip the KG short-circuit and pull every triple
             from context-fill. Use for benchmarks whose answers live
             exclusively in context paragraphs (e.g. MuSiQue)."""
    paragraphs = _normalize_context(context) if context else []
    # For each question predicate, the matcher targets are its grounded
    # kg_ids (multiple — direct OR inverse matches). If ungrounded (no
    # kg_ids), we fall back to the original label for fuzzy matching.
    pred_targets: List[List[str]] = [
        (p.kg_ids if p.kg_ids else [p.label])
        for p in bag.predicates
    ]
    anchor_labels = [e.label for e in bag.entities]
    # Single-predicate queries may need to reuse the same predicate over
    # multiple hops; cap reuse at R_max (default 2). Multi-predicate
    # queries use one hop per predicate (each predicate is consumed once).
    single_predicate_reuse = (len(bag.predicates) == 1)
    _rmax = int(os.environ.get("UNFOLD_RMAX", os.environ.get("UNFOLD_MAX_TRANS_HOPS", "2")))
    max_hops = _rmax if single_predicate_reuse else max(len(bag.predicates), 1)

    # Initial states: one per anchor, empty consumed set, origin = self.
    states: Set[State] = {
        (e.label, frozenset(), e.label) for e in bag.entities
    }
    visited_states: Set[State] = set(states)
    # Hop at which each state was first reached. Anchors are at hop 0.
    state_hop: Dict[State, int] = {st: 0 for st in states}
    # Multi-parent BFS: only edges that advance a state on a minimal-hop path
    # are kept (no orphan triples).
    edges_collected: Set[Tuple[str, str, str]] = set()

    for hop in range(max_hops):
        if verbose:
            print(f"  [hop {hop}] {len(states)} states")
        # Per-entity union of predicate indices that any live state still has.
        entity_to_remaining: Dict[str, Set[int]] = {}
        state_remaining: Dict[State, List[int]] = {}
        for st in states:
            ent, consumed, _origin = st
            if single_predicate_reuse:
                remaining = [0]
            else:
                remaining = [i for i in range(len(bag.predicates)) if i not in consumed]
            if not remaining:
                continue
            state_remaining[st] = remaining
            entity_to_remaining.setdefault(ent, set()).update(remaining)
        if not state_remaining:
            break

        # Phase 1: KG short-circuit. Pull neighborhood (predicate-filtered at SQL
        # level so we don't lose relevant edges to dense-anchor caps) and accumulate.
        # Skipped entirely when skip_kg=True (e.g. MuSiQue, context-only).
        entity_kg_triples: Dict[str, List[Tuple[str, str, str]]] = {}
        entity_kg_covered_preds: Dict[str, Set[int]] = {}
        if not skip_kg:
            for ent in entity_to_remaining:
                needed_targets: List[str] = []
                for pi in entity_to_remaining[ent]:
                    needed_targets.extend(pred_targets[pi])
                if not needed_targets:
                    continue
                kg_filtered = _kg_neighbors_for_predicates(ent, needed_targets)
                if not kg_filtered:
                    continue
                entity_kg_triples[ent] = kg_filtered
                covered = set()
                for s, p, o in kg_filtered:
                    for pi in entity_to_remaining[ent]:
                        if any(_pred_matches(p, t) for t in pred_targets[pi]):
                            covered.add(pi)
                            break
                entity_kg_covered_preds[ent] = covered

        # Phase 2: context fill for KG misses. ONE LLM call per entity, run
        # IN PARALLEL across entities for the same hop. Each call asks for
        # all of an entity's missing predicates at once.
        entity_ctx_triples: Dict[str, List[Tuple[str, str, str]]] = {}
        if paragraphs:
            jobs = []
            for ent, remaining_set in entity_to_remaining.items():
                covered = entity_kg_covered_preds.get(ent, set())
                missing = [pi for pi in remaining_set if pi not in covered]
                if not missing:
                    continue
                preds_to_ask = [
                    {"label": bag.predicates[pi].label,
                     "type":  bag.predicates[pi].type}
                    for pi in missing
                ]
                jobs.append((ent, preds_to_ask))

            if jobs:
                def _do_fill(ent, preds_to_ask):
                    return ent, _batched_context_fill(
                        ent, preds_to_ask, paragraphs, client, model
                    )
                # Run all entities' fills concurrently.
                with ThreadPoolExecutor(max_workers=min(len(jobs), 8)) as pool:
                    futures = [pool.submit(_do_fill, ent, preds)
                               for ent, preds in jobs]
                    for fut in as_completed(futures):
                        ent, triples = fut.result()
                        if triples:
                            entity_ctx_triples[ent] = triples

        # Advance states. An edge is kept iff it advances a state on a
        # minimal-hop path (multi-parent: same-hop rediscovery from a
        # different parent is also a minimal path and is kept).
        new_states: Set[State] = set()
        next_hop = hop + 1
        for st, remaining in state_remaining.items():
            ent, consumed, origin = st
            kg_t = entity_kg_triples.get(ent, [])
            ctx_t = entity_ctx_triples.get(ent, [])
            for triple in (kg_t + ctx_t):
                # Determine which q_pred index this triple's predicate matches.
                matched_pi = None
                for pi in remaining:
                    if any(_pred_matches(triple[1], t) for t in pred_targets[pi]):
                        matched_pi = pi
                        break
                if matched_pi is None:
                    continue
                # Advance the state along this edge. In single-predicate
                # reuse mode, don't consume the predicate (allow chained reuse
                # up to R_max hops).
                next_ent = _other_endpoint(triple, ent)
                if next_ent.lower().strip() == ent.lower().strip():
                    continue  # self-loop
                new_consumed = consumed if single_predicate_reuse else (consumed | {matched_pi})
                new_state: State = (next_ent, new_consumed, origin)
                if new_state not in visited_states:
                    visited_states.add(new_state)
                    state_hop[new_state] = next_hop
                    new_states.add(new_state)
                    edges_collected.add(triple)
                elif state_hop.get(new_state) == next_hop:
                    # Same-hop rediscovery from a different parent — also minimal.
                    edges_collected.add(triple)
                # else: non-minimal rediscovery, skip the edge.

        if not new_states:
            break
        states = new_states

    return list(edges_collected)


# ============================================================
# WALK: rules that refine the subgraph into a smaller, organized form
#       (currently absorbed into POSE via the LLM compose call;
#        a deterministic refinement layer can replace this later)
# ============================================================


# ============================================================
# POSE: final LLM call producing the answer string
# ============================================================

_COMPOSE_PROMPT = """You answer a question using ONLY the retrieved facts.

Output the answer DIRECTLY. No reasoning, no preamble, no markdown, no \
"Final answer:" prefix.
  - For yes/no questions: respond with exactly "YES" or "NO".
  - For factoid questions: respond with a short phrase, entity, or value \
    (no period at the end, no surrounding quotes).
"""




def _extract_final_answer(text: str) -> str:
    """Trim the model's direct-answer output."""
    if not text:
        return ""
    return text.strip()


_SYSTEM_YESNO = "Answer YES or NO."
_SYSTEM_FINAL = (
    "Consider the question's structure and answer based on the resolved "
    "facts. Output only the answer — YES/NO for yes/no questions, a short "
    "phrase, entity, or value otherwise. No explanation, no prefix."
)


def _edge_to_sentence(triple: Tuple[str, str, str]) -> str:
    s, p, o = triple
    pred = str(p).strip().replace("_", " ")
    return f"{s} {pred} {o}"


def _build_subqueries_from_subgraph(
        edges: List[Tuple[str, str, str]], max_sqs: int = 10
        ) -> List[Tuple[str, Tuple[str, str, str]]]:
    """One SQ per unique edge in BFS order, capped at max_sqs."""
    seen = set()
    sqs: List[Tuple[str, Tuple[str, str, str]]] = []
    for triple in edges:
        s, p, o = triple
        key = (str(s).lower().strip(), str(p).lower().strip(), str(o).lower().strip())
        if key in seen:
            continue
        seen.add(key)
        sentence = _edge_to_sentence(triple)
        sqs.append((f"is it true that {sentence}?", triple))
        if len(sqs) >= max_sqs:
            break
    return sqs


def _norm_for_match(s: str) -> str:
    return ' '.join(re.sub(r'[^a-z0-9\s]', ' ', str(s).lower()).split())


_LEAF_PRUNE_SYSTEM = (
    "You judge which candidate is the most plausible value for a given relation. "
    "Reply with the single most plausible candidate, exactly as it appears in the "
    "list. If two or more candidates are equally plausible and you cannot pick "
    "one, list them separated by ' ;; ' (double semicolons), in order of "
    "decreasing plausibility. Output the candidate string(s) only — no "
    "explanation, no bullets, no quotes."
)


def _llm_pick_best_leaf(predicate: str,
                          candidates: List[str],
                          client, model: str,
                          predicate_type: str = "") -> List[str]:
    """Ask the LLM which candidate(s) best fit `predicate`. Returns a subset
    of `candidates` (always at least one). Falls back to the full input list
    on any failure — safer to under-prune than to drop the correct leaf.

    `predicate_type` is an optional generic type tag (e.g. person, place,
    institution) describing what kind of object the predicate should land
    on. When set, it's surfaced to the LLM as a hint."""
    if not candidates:
        return []
    if len(candidates) == 1:
        return list(candidates)
    if predicate_type:
        pred_field = f"{predicate} (expected type: {predicate_type})"
    else:
        pred_field = predicate
    user_msg = (
        f"Predicate: {pred_field}\n"
        "Candidates:\n" + "\n".join(f"- {c}" for c in candidates)
    )
    try:
        resp = client.chat.completions.create(
            model=model, temperature=_LLM_TEMPERATURE, max_tokens=120,
            messages=[
                {"role": "system", "content": _LEAF_PRUNE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        return list(candidates)
    if not raw:
        return list(candidates)

    parts = [p.strip() for p in raw.split(";;") if p.strip()]
    parts = [re.sub(r'^\s*[-*]\s*', '', p) for p in parts]
    parts = [p.strip('"').strip("'") for p in parts]

    cand_lower = {c.lower().strip(): c for c in candidates}
    picks: List[str] = []
    for p in parts:
        pl = p.lower().strip()
        if pl in cand_lower:
            c = cand_lower[pl]
            if c not in picks:
                picks.append(c)
            continue
        for cl, c in cand_lower.items():
            if cl and (cl in pl or pl in cl):
                if c not in picks:
                    picks.append(c)
                break
    return picks if picks else list(candidates)


def _resolve_predicate_type(pred: str, predicate_types: Dict[str, str]) -> str:
    """Best-effort lookup of a generic type tag for an edge predicate. Tries
    exact (case-insensitive) match against the question's typed predicates,
    falls back to a fuzzy substring/token-overlap match. Returns "" when no
    confident type can be found."""
    if not predicate_types or not pred:
        return ""
    p_lower = pred.lower().strip()
    if p_lower in predicate_types:
        return predicate_types[p_lower]
    p_tokens = set(re.findall(r"\w+", p_lower))
    if not p_tokens:
        return ""
    best = ("", 0.0)
    for k, v in predicate_types.items():
        k_tokens = set(re.findall(r"\w+", k.lower()))
        if not k_tokens:
            continue
        overlap = len(p_tokens & k_tokens) / max(len(p_tokens), len(k_tokens))
        if overlap > best[1]:
            best = (v, overlap)
    return best[0] if best[1] >= 0.5 else ""


def _relevance_trim_single_anchor(edges: List[Tuple[str, str, str]],
                                     anchor_label: str,
                                     client, model: str,
                                     verbose: bool = False,
                                     predicate_types: Optional[Dict[str, str]] = None
                                     ) -> Tuple[List[Tuple[str, str, str]],
                                                List[Tuple[str, str, str]]]:
    """Leaf-level LLM-guided trim of the forward tree from a single anchor.

    Walks the forward (subject→object) BFS tree from `anchor_label`, finds the
    leaves (nodes with no outgoing forward edge inside the tree), groups them
    by the predicate of their incoming edge, and asks the LLM per group which
    leaf is most plausible. Surviving leaves are propagated upward: any edge
    on a forward path from anchor to a surviving leaf is kept; the rest are
    moved to the ignored set (still returned for transparency).

    Edges not in the forward tree at all (e.g. context-fill edges discovered
    from the other direction) pass through to `kept` unchanged.

    Returns (kept_edges, ignored_edges)."""
    edges = list(edges)
    if not edges:
        return [], []

    adj_fwd: Dict[str, List[Tuple[str, Tuple[str, str, str]]]] = {}
    adj_rev: Dict[str, List[Tuple[str, Tuple[str, str, str]]]] = {}
    norm_to_display: Dict[str, str] = {}
    for triple in edges:
        s, _p, o = triple
        sn, on = _norm_for_match(s), _norm_for_match(o)
        adj_fwd.setdefault(sn, []).append((on, triple))
        adj_rev.setdefault(on, []).append((sn, triple))
        norm_to_display.setdefault(sn, s)
        norm_to_display.setdefault(on, o)

    anchor_n = _norm_for_match(anchor_label)
    if anchor_n not in adj_fwd:
        return list(edges), []

    reachable_nodes: Set[str] = {anchor_n}
    tree_edges: Set[Tuple[str, str, str]] = set()
    q = deque([anchor_n])
    while q:
        u = q.popleft()
        for v, edge in adj_fwd.get(u, []):
            tree_edges.add(edge)
            if v not in reachable_nodes:
                reachable_nodes.add(v)
                q.append(v)

    if not tree_edges:
        return list(edges), []

    leaves: Set[str] = set()
    for n in reachable_nodes:
        if n == anchor_n:
            continue
        has_forward = any(v in reachable_nodes for v, _e in adj_fwd.get(n, []))
        if not has_forward:
            leaves.add(n)

    if not leaves:
        return list(edges), []

    leaves_by_pred: Dict[str, Set[str]] = {}
    for leaf_n in leaves:
        for parent_n, edge in adj_rev.get(leaf_n, []):
            if parent_n in reachable_nodes:
                _s, p, _o = edge
                leaves_by_pred.setdefault(p, set()).add(leaf_n)

    surviving_leaves: Set[str] = set()
    for pred, leaf_set in leaves_by_pred.items():
        if len(leaf_set) <= 1:
            surviving_leaves.update(leaf_set)
            continue
        candidates_display = [norm_to_display[n] for n in sorted(leaf_set)]
        ptype = _resolve_predicate_type(pred, predicate_types or {})
        picks_display = _llm_pick_best_leaf(pred, candidates_display,
                                             client, model,
                                             predicate_type=ptype)
        picks_norm = {_norm_for_match(p) for p in picks_display}
        surviving = leaf_set & picks_norm
        if not surviving:
            surviving = leaf_set  # parse failed → no prune
        surviving_leaves.update(surviving)
        if verbose:
            print(f"  [trim] predicate={pred!r}: {len(leaf_set)} candidates "
                  f"→ kept {len(surviving)} "
                  f"({[norm_to_display[n] for n in sorted(surviving)]})")

    marked: Set[str] = set(surviving_leaves)
    queue = deque(surviving_leaves)
    while queue:
        v = queue.popleft()
        for u, _e in adj_rev.get(v, []):
            if u in reachable_nodes and u not in marked:
                marked.add(u)
                queue.append(u)
    marked.add(anchor_n)

    kept: List[Tuple[str, str, str]] = []
    ignored: List[Tuple[str, str, str]] = []
    for triple in edges:
        if triple in tree_edges:
            s, _p, o = triple
            sn, on = _norm_for_match(s), _norm_for_match(o)
            if sn in marked and on in marked:
                kept.append(triple)
            else:
                ignored.append(triple)
        else:
            kept.append(triple)
    return kept, ignored


def _select_compose_subgraph(edges: List[Tuple[str, str, str]],
                                anchor_labels: List[str],
                                client=None,
                                model: Optional[str] = None,
                                verbose: bool = False,
                                predicate_types: Optional[Dict[str, str]] = None,
                                predicate_labels: Optional[List[str]] = None,
                                ) -> Tuple[List[Tuple[str, str, str]],
                                           List[Tuple[str, str, str]]]:
    """Filter the subgraph to focus on anchor-relevant evidence.

    Returns (kept_edges, ignored_edges).

    - 2+ anchors all pairwise-connected → Steiner spine + predicate-coverage
      extension: after the shortest-path spine is built, any bag predicate
      not yet represented in the spine triggers a one-hop extension — every
      edge with that predicate that touches a spine node is added back. Then
      a per-predicate LLM plausibility trim runs on the extension only
      (spine edges are always preserved to keep the bridge intact).
    - 1 anchor + client/model provided → LLM leaf-trim of the forward tree.
    - Otherwise → plain forward-tree fallback (no trim).
    """
    from collections import deque
    from itertools import combinations
    if not edges or not anchor_labels:
        return list(edges), []
    adj_und: Dict[str, List[Tuple[str, Tuple[str, str, str]]]] = {}
    adj_fwd: Dict[str, List[Tuple[str, Tuple[str, str, str]]]] = {}
    for triple in edges:
        s, _p, o = triple
        sn, on = _norm_for_match(s), _norm_for_match(o)
        adj_und.setdefault(sn, []).append((on, triple))
        adj_und.setdefault(on, []).append((sn, triple))
        adj_fwd.setdefault(sn, []).append((on, triple))
    anchors_n = [_norm_for_match(a) for a in anchor_labels]
    valid = [a for a in anchors_n if a in adj_und]

    def _shortest_path_edges(start: str, target: str):
        if start == target:
            return set()
        parent = {start: None}
        parent_edge: Dict[str, Tuple[str, str, str]] = {}
        q = deque([start])
        found = False
        while q:
            u = q.popleft()
            if u == target:
                found = True; break
            for v, edge in adj_und.get(u, []):
                if v not in parent:
                    parent[v] = u; parent_edge[v] = edge
                    q.append(v)
        if not found:
            return None
        out = set()
        cur = target
        while parent[cur] is not None:
            out.add(parent_edge[cur]); cur = parent[cur]
        return out

    if len(valid) >= 2:
        path_edges = set()
        all_connected = True
        for a, b in combinations(valid, 2):
            p = _shortest_path_edges(a, b)
            if p is None:
                all_connected = False; break
            path_edges.update(p)
        if all_connected and path_edges:
            spine = set(path_edges)
            spine_nodes: Set[str] = set()
            spine_preds: Set[str] = set()
            for s, p, o in spine:
                spine_nodes.add(_norm_for_match(s))
                spine_nodes.add(_norm_for_match(o))
                spine_preds.add(p.lower().strip())
            bag_preds = {(pl or "").lower().strip()
                          for pl in (predicate_labels or [])
                          if (pl or "").strip()}
            missing_preds = bag_preds - spine_preds
            extension: List[Tuple[str, str, str]] = []
            if missing_preds:
                for triple in edges:
                    if triple in spine:
                        continue
                    s, p, o = triple
                    if p.lower().strip() not in missing_preds:
                        continue
                    if (_norm_for_match(s) in spine_nodes
                            or _norm_for_match(o) in spine_nodes):
                        extension.append(triple)
            ext_kept: List[Tuple[str, str, str]] = []
            ext_ignored: List[Tuple[str, str, str]] = []
            if extension and client is not None and model:
                by_pred: Dict[str, List[Tuple[str, str, str]]] = {}
                for t in extension:
                    by_pred.setdefault(t[1], []).append(t)
                for pred, group in by_pred.items():
                    if len(group) <= 1:
                        ext_kept.extend(group)
                        continue
                    cands: List[str] = []
                    seen: Set[str] = set()
                    for s, _p, o in group:
                        for end in (o, s):
                            en = _norm_for_match(end)
                            if en in spine_nodes:
                                continue
                            if en not in seen:
                                seen.add(en)
                                cands.append(end)
                    if len(cands) <= 1:
                        ext_kept.extend(group)
                        continue
                    ptype = _resolve_predicate_type(pred, predicate_types or {})
                    picks = _llm_pick_best_leaf(pred, cands, client, model,
                                                  predicate_type=ptype)
                    picks_norm = {_norm_for_match(x) for x in picks}
                    for t in group:
                        sn = _norm_for_match(t[0])
                        on = _norm_for_match(t[2])
                        cand_end = on if sn in spine_nodes else sn
                        if cand_end in picks_norm:
                            ext_kept.append(t)
                        else:
                            ext_ignored.append(t)
                    if verbose:
                        print(f"  [ext-trim] predicate={pred!r}: {len(group)} "
                              f"candidates → kept {sum(1 for t in group if t in ext_kept)}")
            else:
                ext_kept = list(extension)
            kept = list(spine) + ext_kept
            kept_set = set(kept)
            ignored = [e for e in edges if e not in kept_set]
            if verbose and (extension or ignored):
                print(f"  [select] spine={len(spine)} ext={len(ext_kept)} "
                      f"ignored={len(ignored)} missing_preds={sorted(missing_preds)}")
            return kept, ignored

    # Single-anchor (or multi-anchor not connected): forward-tree fallback.
    # When client/model are available, run LLM leaf-trim against the single
    # anchor that's actually present in the subgraph.
    if len(valid) == 1 and client is not None and model:
        anchor_display = next(a for a in anchor_labels
                              if _norm_for_match(a) == valid[0])
        return _relevance_trim_single_anchor(edges, anchor_display,
                                              client, model, verbose=verbose,
                                              predicate_types=predicate_types)

    tree_edges: Set[Tuple[str, str, str]] = set()
    for a in valid:
        seen = {a}
        q = deque([a])
        while q:
            u = q.popleft()
            for v, edge in adj_fwd.get(u, []):
                if v not in seen:
                    seen.add(v)
                    tree_edges.add(edge)
                    q.append(v)
    return list(tree_edges), []


def _format_resolved_block(prior_qa: List[Tuple[str, str, Tuple[str, str, str]]]
                            ) -> str:
    if not prior_qa:
        return ""
    lines = ["previously resolved:"]
    for i, (sq, ans, _triple) in enumerate(prior_qa):
        sq_idx = i + 1
        if ans == "YES":
            lines.append(f"Yes: SQ{sq_idx}: {sq}")
        elif ans == "NO":
            lines.append(f"No: SQ{sq_idx}: {sq}")
        else:
            lines.append(f"{ans}: SQ{sq_idx}: {sq}")
    lines.append("")
    return "\n".join(lines)


def _extract_yesno(text: str) -> str:
    m = re.findall(r"\b(YES|NO)\b", text or "", re.IGNORECASE)
    return m[-1].upper() if m else "FAIL"


def compose(question: str, edges: List[Tuple[str, str, str]],
             client, model: str, max_sqs: int = 10,
             verbose: bool = False) -> str:
    """POSE: iterative sub-query interaction over a pre-selected edge set.

    Walk the edges as yes/no SQs, each prompt accumulating a
    `previously resolved:` block. Final call asks the original question over
    resolved facts. Subgraph selection (Steiner / forward-tree / LLM trim) is
    expected to have run BEFORE calling this — see _select_compose_subgraph.
    """
    # Tuple-ify in case edges were loaded from JSON (where they'd be lists);
    # downstream set operations require hashable triples.
    edges = [tuple(e) if not isinstance(e, tuple) else e for e in edges]
    sqs = _build_subqueries_from_subgraph(edges, max_sqs=max_sqs)

    # No subgraph evidence — fall through to a single-call best-effort answer.
    if not sqs:
        try:
            resp = client.chat.completions.create(
                model=model, temperature=_COMPOSE_TEMPERATURE, max_tokens=200,
                messages=[
                    {"role": "system", "content": _SYSTEM_FINAL},
                    {"role": "user", "content": f"Main Question: {question}\n"},
                ],
            )
            return _extract_final_answer((resp.choices[0].message.content or "").strip())
        except Exception as e:
            return f"[COMPOSE ERROR: {e}]"

    prior_qa: List[Tuple[str, str, Tuple[str, str, str]]] = []

    for i, (sq, triple) in enumerate(sqs):
        sq_idx = i + 1
        resolved = _format_resolved_block(prior_qa)
        body = f"Main Question: {question}\n\n"
        if resolved:
            body += resolved + "\n"
        body += f"Yes/No SQ{sq_idx}: {sq}\n"
        try:
            resp = client.chat.completions.create(
                model=model, temperature=_COMPOSE_TEMPERATURE, max_tokens=20,
                messages=[
                    {"role": "system", "content": _SYSTEM_YESNO},
                    {"role": "user", "content": body},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            raw = f"[ERROR: {e}]"
        ans = _extract_yesno(raw)
        prior_qa.append((sq, ans, triple))
        if verbose:
            print(f"  SQ{sq_idx}: {sq} -> {ans}")

    # Final call: original question over resolved facts.
    resolved = _format_resolved_block(prior_qa)
    body = f"Main Question: {question}\n\n"
    if resolved:
        body += resolved + "\n"
    body += f"{question}\n"
    try:
        resp = client.chat.completions.create(
            model=model, temperature=_COMPOSE_TEMPERATURE, max_tokens=200,
            messages=[
                {"role": "system", "content": _SYSTEM_FINAL},
                {"role": "user", "content": body},
            ],
        )
        return _extract_final_answer((resp.choices[0].message.content or "").strip())
    except Exception as e:
        return f"[COMPOSE ERROR: {e}]"


def unfold(question: str, context=None, client=None, model: str = "",
              verbose: bool = False, skip_compose: bool = False,
              skip_grounding: bool = False, skip_kg: bool = False,
              src: Optional[str] = None
              ) -> Dict[str, Any]:
    """Top-level entry.
    skip_compose:   when True, skip POSE and return the subgraph only.
    skip_grounding: when True, skip Phase-A grounding (no LLM call to map
                    question predicates to KG predicates).
    skip_kg:        when True, also skip the KG short-circuit inside CONSTRUCT,
                    so all triples come from context-fill. Use for benchmarks
                    whose answers live exclusively in context (e.g. MuSiQue)."""
    import time
    t0 = time.time()
    bag = parse_question(question, client, model, context=context)
    if bag is None:
        return {"answer": "", "error": "parse_failed",
                "subgraph": [], "bag": None,
                "pruned_subgraph": [], "ignored_subgraph": [],
                "status": "parse_failed",
                "t_search": round(time.time() - t0, 2), "t_compose": 0.0}

    # Relevance filter: one LLM call reduces the full context to the top-K
    # paragraphs likely to answer the question. Replaces the literal-mention
    # substring filter at the global level (it still runs per-anchor inside
    # context-fill but on a much smaller set).
    if context:
        context = _llm_filter_paragraphs(question, context, client, model,
                                          top_k=5, verbose=verbose)

    if not skip_grounding:
        bag = ground_bag(bag, question, client, model, src=src)
    if verbose:
        print(f"[bag] E={[(e.label, e.kg_id) for e in bag.entities]}")
        print(f"      P={[(p.label, p.kg_ids) for p in bag.predicates]}")
        print(f"      all_grounded={bag.all_grounded}")
    # Anchor-aware coreference annotation: inline-bracket pronouns / acronyms /
    # role-titles that refer to a question anchor. Original text preserved.
    annotated_context = None
    if context and bag.entities:
        annotated_context = _resolve_coreferences(
            context, [e.label for e in bag.entities],
            client, model, verbose=verbose,
        )
        context = annotated_context
    edges = search(bag, context, client, model, verbose=verbose,
                    skip_kg=skip_kg, src=src)
    # Optimization (paper §3, opt. 4): before BiBFS, link anchors that are
    # not yet covered in G by asking the LLM to propose connecting facts.
    # OFF by default; set UNFOLD_LINK_UNCOVERED=1 to enable.
    if os.environ.get("UNFOLD_LINK_UNCOVERED", "0") == "1":
        anchor_labels_search = [e.label for e in (bag.entities or [])]
        new_links = _link_uncovered_entities(
            anchor_labels_search, set(edges), client, model, verbose=verbose)
        if new_links:
            edges = list(dict.fromkeys(list(edges) + new_links))
    t_search = round(time.time() - t0, 2)
    if verbose:
        print(f"[search] {len(edges)} edges  (t_search={t_search}s)")
        for e in edges[:20]:
            print(f"   {e}")
        if len(edges) > 20:
            print(f"   ... +{len(edges)-20} more")
    if skip_compose:
        return {"answer": "", "subgraph": edges, "bag": asdict(bag),
                "pruned_subgraph": list(edges), "ignored_subgraph": [],
                "annotated_context": annotated_context,
                "status": "no_compose",
                "t_search": t_search, "t_compose": 0.0}
    t1 = time.time()
    anchor_labels = [e.label for e in (bag.entities or [])]
    # Build a predicate-label → type map so the leaf-trim pruning can score
    # candidates by expected object type.
    predicate_types = {
        p.label.lower(): p.type for p in (bag.predicates or [])
        if getattr(p, "type", "")
    } or None
    predicate_labels = [p.label for p in (bag.predicates or []) if p.label]
    pruned_edges, ignored_edges = _select_compose_subgraph(
        edges, anchor_labels, client=client, model=model, verbose=verbose,
        predicate_types=predicate_types,
        predicate_labels=predicate_labels,
    )
    if verbose and ignored_edges:
        print(f"[select] kept {len(pruned_edges)} edges, "
              f"ignored {len(ignored_edges)} (LLM leaf-trim)")
    answer = compose(question, pruned_edges, client, model, verbose=verbose)
    t_compose = round(time.time() - t1, 2)
    if verbose:
        print(f"[compose] {answer}  (t_compose={t_compose}s)")
    return {"answer": answer, "subgraph": edges, "bag": asdict(bag),
            "pruned_subgraph": pruned_edges,
            "ignored_subgraph": ignored_edges,
            "annotated_context": annotated_context,
            "status": "ok",
            "t_search": t_search, "t_compose": t_compose}


if __name__ == "__main__":
    # Smoke test
    import os
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"], timeout=60)
    model = "deepseek/deepseek-chat-v3.1"
    questions = [
        "Italy and Slovenia have different currencies?",
        "Where was the director of Inception born?",
        "Who was born first, Alberto Varela or Raymond Birt?",
        "Is Lunar Prospector NOT powered by ion thruster?",
        "Among Argentina, Brazil, and Chile, which has the largest area?",
        "Vine is owned by X Corp.?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        out = unfold(q, client=client, model=model, verbose=True)
        print(f"  answer = {out.get('answer', '')!r}")
        print(f"  pruned_subgraph ({len(out.get('pruned_subgraph', []))} edges used by compose):")
        for e in out.get('pruned_subgraph', [])[:10]:
            print(f"    {e}")
        ig = out.get('ignored_subgraph', [])
        if ig:
            print(f"  ignored_subgraph ({len(ig)} edges kept but not used):")
            for e in ig[:10]:
                print(f"    {e}")
