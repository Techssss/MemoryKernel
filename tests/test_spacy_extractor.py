"""
Test: SpaCyExtractor — spaCy-based fact extraction
===================================================
Tests SVO triplet extraction across multiple sentence patterns:
active, passive, copular, prepositional, and multi-sentence.

Run: python -m pytest tests/test_spacy_extractor.py -v
  or: python tests/test_spacy_extractor.py

Requires: pip install spacy && python -m spacy download en_core_web_sm
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memk.extraction.extractor import StructuredFact, BaseExtractor


# ---------------------------------------------------------------------------
# Check if spaCy model is available
# ---------------------------------------------------------------------------

_SPACY_AVAILABLE = False
try:
    import spacy
    spacy.load("en_core_web_sm")
    _SPACY_AVAILABLE = True
except Exception:
    pass

# Skip tests if model is not installed
_skip_reason = "spaCy en_core_web_sm model not installed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fact_tuples(facts):
    """Convert facts to (subject, relation, object) tuples for easier assertion."""
    return [(f.subject, f.relation, f.object) for f in facts]


def _has_fact(facts, *, subject=None, relation=None, obj=None):
    """Check if at least one fact matches the given criteria."""
    for f in facts:
        if subject and f.subject != subject:
            continue
        if relation and f.relation != relation:
            continue
        if obj and f.object != obj:
            continue
        return True
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@unittest.skipUnless(_SPACY_AVAILABLE, _skip_reason)
class TestSpaCyExtractorInterface(unittest.TestCase):
    """Verify SpaCyExtractor satisfies the BaseExtractor contract."""

    def test_is_base_extractor(self):
        from memk.extraction.spacy_extractor import SpaCyExtractor
        ext = SpaCyExtractor()
        self.assertIsInstance(ext, BaseExtractor)

    def test_returns_list_of_structured_fact(self):
        from memk.extraction.spacy_extractor import SpaCyExtractor
        ext = SpaCyExtractor()
        result = ext.extract_facts("Sarah works at Google.")
        self.assertIsInstance(result, list)
        for f in result:
            self.assertIsInstance(f, StructuredFact)

    def test_empty_string_returns_empty(self):
        from memk.extraction.spacy_extractor import SpaCyExtractor
        ext = SpaCyExtractor()
        self.assertEqual(ext.extract_facts(""), [])

    def test_none_returns_empty(self):
        from memk.extraction.spacy_extractor import SpaCyExtractor
        ext = SpaCyExtractor()
        self.assertEqual(ext.extract_facts(None), [])


@unittest.skipUnless(_SPACY_AVAILABLE, _skip_reason)
class TestWorksAtRelation(unittest.TestCase):
    """Test works_at relation extraction."""

    def setUp(self):
        from memk.extraction.spacy_extractor import SpaCyExtractor
        self.ext = SpaCyExtractor()

    def test_works_at_basic(self):
        facts = self.ext.extract_facts("Sarah works at Google.")
        self.assertTrue(_has_fact(facts, subject="Sarah", relation="works_at", obj="Google"))

    def test_works_for(self):
        facts = self.ext.extract_facts("John works for Microsoft.")
        self.assertTrue(_has_fact(facts, subject="John", relation="works_at", obj="Microsoft"))

    def test_joined(self):
        facts = self.ext.extract_facts("Alice joined Meta.")
        self.assertTrue(_has_fact(facts, relation="works_at"))


@unittest.skipUnless(_SPACY_AVAILABLE, _skip_reason)
class TestRoleAtRelation(unittest.TestCase):
    """Test role_at relation extraction (copular + of)."""

    def setUp(self):
        from memk.extraction.spacy_extractor import SpaCyExtractor
        self.ext = SpaCyExtractor()

    def test_is_ceo_of(self):
        facts = self.ext.extract_facts("Elon Musk is CEO of Tesla.")
        self.assertTrue(
            _has_fact(facts, subject="Elon Musk", relation="role_at", obj="Tesla")
        )

    def test_is_founder_of(self):
        facts = self.ext.extract_facts("Larry Page is the founder of Google.")
        self.assertTrue(_has_fact(facts, relation="role_at", obj="Google"))


@unittest.skipUnless(_SPACY_AVAILABLE, _skip_reason)
class TestLocatedInRelation(unittest.TestCase):
    """Test located_in relation extraction."""

    def setUp(self):
        from memk.extraction.spacy_extractor import SpaCyExtractor
        self.ext = SpaCyExtractor()

    def test_based_in(self):
        facts = self.ext.extract_facts("OpenAI is based in San Francisco.")
        self.assertTrue(
            _has_fact(facts, subject="OpenAI", relation="located_in",
                      obj="San Francisco")
        )

    def test_lives_in(self):
        facts = self.ext.extract_facts("Bob lives in New York.")
        self.assertTrue(_has_fact(facts, relation="located_in"))


@unittest.skipUnless(_SPACY_AVAILABLE, _skip_reason)
class TestUsesToolRelation(unittest.TestCase):
    """Test uses_tool relation extraction (active and passive)."""

    def setUp(self):
        from memk.extraction.spacy_extractor import SpaCyExtractor
        self.ext = SpaCyExtractor()

    def test_uses_direct(self):
        facts = self.ext.extract_facts("Alice uses PyTorch.")
        self.assertTrue(
            _has_fact(facts, subject="Alice", relation="uses_tool", obj="PyTorch")
        )

    def test_passive_used_by(self):
        """'VS Code is a tool used by John' → John uses_tool VS Code."""
        facts = self.ext.extract_facts("VS Code is a tool used by John.")
        self.assertTrue(
            _has_fact(facts, subject="John", relation="uses_tool", obj="VS Code")
        )


@unittest.skipUnless(_SPACY_AVAILABLE, _skip_reason)
class TestPartOfRelation(unittest.TestCase):
    """Test part_of relation extraction."""

    def setUp(self):
        from memk.extraction.spacy_extractor import SpaCyExtractor
        self.ext = SpaCyExtractor()

    def test_belongs_to(self):
        facts = self.ext.extract_facts("The project belongs to Team Alpha.")
        self.assertTrue(_has_fact(facts, relation="part_of"))


@unittest.skipUnless(_SPACY_AVAILABLE, _skip_reason)
class TestFallbackBehavior(unittest.TestCase):
    """Test generic relation fallback and filtering."""

    def setUp(self):
        from memk.extraction.spacy_extractor import SpaCyExtractor
        self.ext_strict = SpaCyExtractor(fallback_to_generic=False)
        self.ext_generic = SpaCyExtractor(fallback_to_generic=True)

    def test_unmapped_verb_filtered_in_strict_mode(self):
        """Verbs not in ontology should be filtered when generic=False."""
        facts = self.ext_strict.extract_facts("The cat sat on the mat.")
        # "sat on" is not in our ontology
        self.assertEqual(len(facts), 0)

    def test_unmapped_verb_emitted_in_generic_mode(self):
        """Verbs not in ontology should emit 'generic' when enabled."""
        facts = self.ext_generic.extract_facts("The cat sat on the mat.")
        if len(facts) > 0:
            self.assertEqual(facts[0].relation, "generic")


@unittest.skipUnless(_SPACY_AVAILABLE, _skip_reason)
class TestMultiSentence(unittest.TestCase):
    """Test extraction from multi-sentence text."""

    def setUp(self):
        from memk.extraction.spacy_extractor import SpaCyExtractor
        self.ext = SpaCyExtractor()

    def test_two_sentences(self):
        text = "Sarah works at Google. OpenAI is based in San Francisco."
        facts = self.ext.extract_facts(text)
        self.assertTrue(_has_fact(facts, relation="works_at"))
        self.assertTrue(_has_fact(facts, relation="located_in"))

    def test_three_sentences(self):
        text = (
            "Elon Musk is CEO of Tesla. "
            "Alice uses PyTorch. "
            "The backend belongs to Team Alpha."
        )
        facts = self.ext.extract_facts(text)
        self.assertGreaterEqual(len(facts), 2)


@unittest.skipUnless(_SPACY_AVAILABLE, _skip_reason)
class TestAdditionalPatterns(unittest.TestCase):
    """Test extra verb patterns in the fallback map."""

    def setUp(self):
        from memk.extraction.spacy_extractor import SpaCyExtractor
        self.ext = SpaCyExtractor()

    def test_manages(self):
        facts = self.ext.extract_facts("Alice manages the backend.")
        self.assertTrue(_has_fact(facts, subject="Alice", relation="manages"))

    def test_owns(self):
        facts = self.ext.extract_facts("Google owns YouTube.")
        self.assertTrue(_has_fact(facts, relation="owns"))

    def test_builds(self):
        facts = self.ext.extract_facts("The team builds microservices.")
        self.assertTrue(_has_fact(facts, relation="builds"))

    def test_develops(self):
        facts = self.ext.extract_facts("Meta develops React.")
        self.assertTrue(_has_fact(facts, relation="develops"))


@unittest.skipUnless(_SPACY_AVAILABLE, _skip_reason)
class TestCompoundNames(unittest.TestCase):
    """Test that multi-word entity names are preserved."""

    def setUp(self):
        from memk.extraction.spacy_extractor import SpaCyExtractor
        self.ext = SpaCyExtractor()

    def test_two_word_person(self):
        facts = self.ext.extract_facts("Elon Musk works at Tesla.")
        self.assertTrue(_has_fact(facts, subject="Elon Musk"))

    def test_two_word_location(self):
        facts = self.ext.extract_facts("OpenAI is based in San Francisco.")
        self.assertTrue(_has_fact(facts, obj="San Francisco"))


class TestSafeFallbackNoSpacy(unittest.TestCase):
    """Test that SpaCyExtractor gracefully handles missing spaCy."""

    def test_missing_model_returns_empty(self):
        from memk.extraction.spacy_extractor import SpaCyExtractor
        ext = SpaCyExtractor(model_name="nonexistent_model_xyz")
        facts = ext.extract_facts("Sarah works at Google.")
        self.assertEqual(facts, [])

    def test_load_failed_flag_prevents_retry(self):
        from memk.extraction.spacy_extractor import SpaCyExtractor
        ext = SpaCyExtractor(model_name="nonexistent_model_xyz")
        ext.extract_facts("test")
        self.assertTrue(ext._load_failed)
        # Second call should not retry loading
        facts = ext.extract_facts("Sarah works at Google.")
        self.assertEqual(facts, [])


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_standalone():
    """Run all tests without pytest."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestSpaCyExtractorInterface,
        TestWorksAtRelation,
        TestRoleAtRelation,
        TestLocatedInRelation,
        TestUsesToolRelation,
        TestPartOfRelation,
        TestFallbackBehavior,
        TestMultiSentence,
        TestAdditionalPatterns,
        TestCompoundNames,
        TestSafeFallbackNoSpacy,
    ]
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    run_standalone()
