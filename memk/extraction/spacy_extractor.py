"""
memk.extraction.spacy_extractor
================================
SpaCy-based fact extractor — lightweight, rule-first approach.

Uses spaCy's dependency parse to find Subject-Verb-Object patterns
and maps them to a narrow ontology of relation types:

    works_at, role_at, located_in, part_of, uses_tool, alias_of, generic

Safe fallback: if ``en_core_web_sm`` is not installed, ``extract_facts()``
returns an empty list and logs a warning (no crash).

Setup
-----
::

    pip install spacy
    python -m spacy download en_core_web_sm
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from memk.extraction.extractor import BaseExtractor, StructuredFact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Relation ontology mapping
# ---------------------------------------------------------------------------

# Verb lemma + preposition → canonical relation
_VERB_PREP_MAP: Dict[Tuple[str, Optional[str]], str] = {
    # works_at
    ("work", "at"):        "works_at",
    ("work", "for"):       "works_at",
    ("employ", "at"):      "works_at",
    ("employ", "by"):      "works_at",
    ("join", None):        "works_at",
    ("join", "at"):        "works_at",
    # role_at
    ("be", "of"):          "role_at",      # "X is CEO of Y"
    ("serve", "as"):       "role_at",
    ("lead", None):        "role_at",
    # located_in
    ("base", "in"):        "located_in",   # "X is based in Y"
    ("locate", "in"):      "located_in",
    ("live", "in"):        "located_in",
    ("reside", "in"):      "located_in",
    ("headquarter", "in"): "located_in",
    ("situate", "in"):     "located_in",
    # part_of
    ("belong", "to"):      "part_of",
    ("be", "part"):        "part_of",      # "X is part of Y"
    # uses_tool
    ("use", None):         "uses_tool",
    ("use", "by"):         "uses_tool",    # passive: "X is used by Y"
    ("utilize", None):     "uses_tool",
    ("adopt", None):       "uses_tool",
    # alias_of
    ("know", "as"):        "alias_of",     # "X is known as Y"
    ("call", None):        "alias_of",
    ("rename", "to"):      "alias_of",
}

# Broader verb lemma → relation (no preposition required)
_VERB_FALLBACK_MAP: Dict[str, str] = {
    "work":   "works_at",
    "employ": "works_at",
    "live":   "located_in",
    "use":    "uses_tool",
    "manage": "manages",
    "own":    "owns",
    "build":  "builds",
    "create": "builds",
    "found":  "founded",
    "develop": "develops",
}

# Prepositions that commonly introduce objects
_OBJ_PREPS: Set[str] = {"at", "for", "in", "of", "by", "to", "with", "as", "from"}


# ---------------------------------------------------------------------------
# SpaCy subtree helpers
# ---------------------------------------------------------------------------

def _get_span_text(token) -> str:
    """Get the full noun phrase span for a token using its subtree."""
    subtree = sorted(token.subtree, key=lambda t: t.i)
    # Filter out punct, conj markers, and relative clauses
    keep = [t for t in subtree
            if t.dep_ not in ("punct", "cc", "relcl", "advcl", "mark")
            and t.pos_ != "PUNCT"]
    if not keep:
        return token.text
    return " ".join(t.text for t in keep)


def _get_compact_span(token) -> str:
    """
    Get a compact noun phrase — the token plus its immediate compounds
    and modifiers, but not deeply nested clauses.
    """
    parts = [token]
    for child in token.children:
        if child.dep_ in ("compound", "amod", "nmod", "flat", "flat:name"):
            parts.append(child)
            # Include grandchild compounds (e.g., "San Francisco")
            for gc in child.children:
                if gc.dep_ in ("compound", "flat", "flat:name"):
                    parts.append(gc)
    parts.sort(key=lambda t: t.i)
    return " ".join(t.text for t in parts)


def _find_prep_obj(token) -> Tuple[Optional[str], Optional[str]]:
    """
    Find the first prepositional object attached to a verb/root token.

    Also matches ``agent`` dependency (passive "by" construction).
    Returns (preposition_text, object_span_text) or (None, None).
    """
    for child in token.children:
        if (child.dep_ in ("prep", "agent")
                and child.text.lower() in _OBJ_PREPS):
            for pobj in child.children:
                if pobj.dep_ in ("pobj", "pcomp"):
                    return child.text.lower(), _get_compact_span(pobj)
    return None, None


def _find_attr_obj(token) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Handle copular constructions: "X is CEO of Y", "X is a tool used by Y".

    Returns
    -------
    (preposition, object_text, acl_verb_lemma_or_None)
        acl_verb_lemma is set when the object comes from a participial
        modifier (acl/relcl), e.g. "used" in "a tool used by John".
        The caller should use acl_verb_lemma (not the root verb) for
        relation resolution in that case.
    """
    for child in token.children:
        if child.dep_ == "attr":
            # e.g., "is CEO" → look at "of Tesla"
            for attr_child in child.children:
                if attr_child.dep_ == "prep" and attr_child.text.lower() in _OBJ_PREPS:
                    for pobj in attr_child.children:
                        if pobj.dep_ in ("pobj", "pcomp"):
                            return attr_child.text.lower(), _get_compact_span(pobj), None
                # Handle participial modifiers: "a tool used by John"
                if attr_child.dep_ in ("acl", "relcl"):
                    prep_text, pobj_text = _find_prep_obj(attr_child)
                    if pobj_text:
                        return prep_text, pobj_text, attr_child.lemma_.lower()
    return None, None, None


def _resolve_relation(verb_lemma: str, prep: Optional[str]) -> str:
    """
    Map a (verb_lemma, preposition) pair to a canonical relation.

    Priority:
    1. Exact (lemma, prep) match in _VERB_PREP_MAP
    2. Verb-only fallback in _VERB_FALLBACK_MAP
    3. Return "generic" as a safe default
    """
    # 1. Exact match
    key = (verb_lemma, prep)
    if key in _VERB_PREP_MAP:
        return _VERB_PREP_MAP[key]

    # 2. Verb-only fallback
    if verb_lemma in _VERB_FALLBACK_MAP:
        return _VERB_FALLBACK_MAP[verb_lemma]

    # 3. Safe generic
    return "generic"


# ---------------------------------------------------------------------------
# SpaCyExtractor
# ---------------------------------------------------------------------------

class SpaCyExtractor(BaseExtractor):
    """
    Fact extractor using spaCy dependency parsing.

    Extracts Subject-Predicate-Object triplets from English text by:
    1. Splitting text into sentences (spaCy sentencizer)
    2. Finding the ROOT verb of each sentence
    3. Extracting subject (nsubj/nsubjpass) and object (dobj/pobj/attr)
    4. Mapping the verb+prep to a narrow ontology relation

    Parameters
    ----------
    model_name : str
        spaCy model to load (default: ``en_core_web_sm``).
    fallback_to_generic : bool
        If True, emit triplets with ``relation="generic"`` when no
        ontology match is found. If False, skip those sentences.

    Usage
    -----
    >>> ext = SpaCyExtractor()
    >>> facts = ext.extract_facts("Sarah works at Google.")
    >>> facts[0].subject
    'Sarah'
    >>> facts[0].relation
    'works_at'
    >>> facts[0].object
    'Google'

    Note
    ----
    If the spaCy model is not installed, ``extract_facts()`` returns ``[]``
    and logs a warning. No exception is raised.
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        fallback_to_generic: bool = False,
    ):
        self._model_name = model_name
        self._fallback_to_generic = fallback_to_generic
        self._nlp = None           # Lazy-loaded
        self._load_failed = False  # Prevent repeated load attempts

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> bool:
        """
        Load spaCy model on first use. Returns True if model is ready.
        Thread-safe: worst case two threads load simultaneously, both succeed.
        """
        if self._nlp is not None:
            return True
        if self._load_failed:
            return False

        try:
            import spacy
            self._nlp = spacy.load(self._model_name)
            logger.info(
                f"SpaCyExtractor: loaded model '{self._model_name}' "
                f"({self._nlp.meta.get('name', '?')})"
            )
            return True
        except OSError:
            logger.warning(
                f"SpaCyExtractor: model '{self._model_name}' not found. "
                f"Run: python -m spacy download {self._model_name}"
            )
            self._load_failed = True
            return False
        except ImportError:
            logger.warning(
                "SpaCyExtractor: spacy is not installed. "
                "Run: pip install spacy"
            )
            self._load_failed = True
            return False

    # ------------------------------------------------------------------
    # Core extraction logic
    # ------------------------------------------------------------------

    def _extract_from_sentence(self, sent) -> List[StructuredFact]:
        """Extract triplets from a single spaCy Span (sentence)."""
        facts: List[StructuredFact] = []

        root = None
        for token in sent:
            if token.dep_ == "ROOT":
                root = token
                break

        if root is None:
            return facts

        verb_lemma = root.lemma_.lower()

        # --- Find subject ---------------------------------------------------
        subject_token = None
        for child in root.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subject_token = child
                break

        if subject_token is None:
            return facts

        subject_text = _get_compact_span(subject_token)

        # --- Find object (multiple strategies) -------------------------------

        # Strategy 1: Direct object (dobj)
        dobj_text = None
        for child in root.children:
            if child.dep_ == "dobj":
                dobj_text = _get_compact_span(child)
                break

        # Strategy 2: Prepositional object on the root verb
        prep_text, pobj_text = _find_prep_obj(root)

        # Strategy 3: Attribute + prep (copular: "X is CEO of Y")
        attr_prep, attr_pobj, acl_verb = _find_attr_obj(root)

        # --- Decide which pattern to emit -----------------------------------

        # Case A: "X works at Y" → verb + prep + pobj
        if pobj_text and prep_text:
            rel = _resolve_relation(verb_lemma, prep_text)
            if rel != "generic" or self._fallback_to_generic:
                facts.append(StructuredFact(
                    subject=subject_text.strip(),
                    relation=rel,
                    object=pobj_text.strip(),
                ))

        # Case B: Copular with attr prep → "X is CEO of Y"
        #         or participial: "X is a tool used by Y"
        elif attr_pobj and attr_prep:
            # Use acl verb lemma if available (e.g. "use" from "used by")
            resolve_verb = acl_verb if acl_verb else verb_lemma
            rel = _resolve_relation(resolve_verb, attr_prep)

            # Passive flip: "X is a tool used by Y" → "Y uses_tool X"
            if acl_verb and attr_prep == "by" and rel == "uses_tool":
                facts.append(StructuredFact(
                    subject=attr_pobj.strip(),
                    relation=rel,
                    object=subject_text.strip(),
                ))
            elif rel != "generic" or self._fallback_to_generic:
                facts.append(StructuredFact(
                    subject=subject_text.strip(),
                    relation=rel,
                    object=attr_pobj.strip(),
                ))

        # Case C: Simple SVO → "X uses Y"
        elif dobj_text:
            rel = _resolve_relation(verb_lemma, None)
            if rel != "generic" or self._fallback_to_generic:
                facts.append(StructuredFact(
                    subject=subject_text.strip(),
                    relation=rel,
                    object=dobj_text.strip(),
                ))

        # Case D: Passive with agent → "X is used by Y" → flip to "Y uses X"
        if not facts:
            # Check for passive voice with agent
            is_passive = subject_token.dep_ == "nsubjpass"
            if is_passive:
                agent_prep, agent_obj = None, None
                for child in root.children:
                    if child.dep_ == "agent" or (
                        child.dep_ == "prep" and child.text.lower() == "by"
                    ):
                        for pobj_child in child.children:
                            if pobj_child.dep_ in ("pobj", "pcomp"):
                                agent_prep = "by"
                                agent_obj = _get_compact_span(pobj_child)
                                break

                if agent_obj:
                    # Flip: "X is used by Y" → "Y uses X"
                    rel = _resolve_relation(verb_lemma, agent_prep)
                    # For passive "used by", map to uses_tool with flipped args
                    if rel == "uses_tool":
                        facts.append(StructuredFact(
                            subject=agent_obj.strip(),
                            relation=rel,
                            object=subject_text.strip(),
                        ))
                    elif rel != "generic" or self._fallback_to_generic:
                        facts.append(StructuredFact(
                            subject=subject_text.strip(),
                            relation=rel,
                            object=agent_obj.strip(),
                        ))

        return facts

    # ------------------------------------------------------------------
    # Public API — implements BaseExtractor
    # ------------------------------------------------------------------

    def extract_facts(self, text: str) -> List[StructuredFact]:
        """
        Extract structured triplets from raw text.

        Parameters
        ----------
        text : str
            Input text (English). Can contain multiple sentences.

        Returns
        -------
        List[StructuredFact]
            Extracted (subject, relation, object) triplets.
            Returns empty list if spaCy model is unavailable.
        """
        text = (text or "").strip()
        if not text:
            return []

        if not self._ensure_model():
            return []

        doc = self._nlp(text)
        all_facts: List[StructuredFact] = []

        for sent in doc.sents:
            try:
                facts = self._extract_from_sentence(sent)
                all_facts.extend(facts)
            except Exception as e:
                logger.debug(
                    f"SpaCyExtractor: skipped sentence '{sent.text[:50]}': {e}"
                )
                continue

        return all_facts
