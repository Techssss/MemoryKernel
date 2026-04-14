import pytest
from memk.extraction.extractor import RuleBasedExtractor

@pytest.fixture
def extractor():
    return RuleBasedExtractor()

def test_extract_user_likes(extractor):
    text = "User thích Kotlin"
    facts = extractor.extract_facts(text)
    assert len(facts) == 1
    assert facts[0].subject == "user"
    assert facts[0].relation == "likes"
    assert facts[0].object == "Kotlin"

def test_extract_user_dislikes(extractor):
    text = "Tôi ghét callback"
    facts = extractor.extract_facts(text)
    assert len(facts) == 1
    assert facts[0].subject == "user"
    assert facts[0].relation == "dislikes"
    assert facts[0].object == "callback"

def test_extract_project_uses(extractor):
    text = "Project dùng Spring Boot"
    facts = extractor.extract_facts(text)
    assert len(facts) == 1
    assert facts[0].subject == "project"
    assert facts[0].relation == "uses"
    assert facts[0].object == "Spring Boot"

def test_extract_multiple_facts(extractor):
    text = "User thích Kotlin, project dùng Spring Boot. Ồn ào không liên quan."
    facts = extractor.extract_facts(text)
    assert len(facts) == 2
    assert facts[0].subject == "user"
    assert facts[0].relation == "likes"
    assert facts[0].object == "Kotlin"
    assert facts[1].subject == "project"
    assert facts[1].relation == "uses"
    assert facts[1].object == "Spring Boot"

def test_skip_noisy_text(extractor):
    text = "Chào buổi sáng AI. Phân tích đoạn này đi."
    facts = extractor.extract_facts(text)
    assert len(facts) == 0

def test_english_extraction(extractor):
    text = "User uses Typer"
    facts = extractor.extract_facts(text)
    assert len(facts) == 1
    assert facts[0].subject == "user"
    assert facts[0].relation == "uses"
    assert facts[0].object == "Typer"
