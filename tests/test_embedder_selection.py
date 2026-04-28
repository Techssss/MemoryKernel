from memk.core import embedder as embedder_mod


def test_hashing_embedder_can_be_selected_without_model_dependencies(monkeypatch):
    monkeypatch.setenv("MEMK_EMBEDDER", "hashing")
    monkeypatch.setattr(embedder_mod, "_DEFAULT_EMBEDDER", None)
    monkeypatch.setattr(embedder_mod, "_DEFAULT_PIPELINE", None)

    embedder = embedder_mod.get_default_embedder()

    assert isinstance(embedder, embedder_mod.HashingEmbedder)
    assert embedder.dim == 128


def test_auto_falls_back_to_hashing_without_tfidf(monkeypatch):
    class MissingSentence:
        def __init__(self):
            raise ImportError("missing sentence-transformers")

    class UnexpectedTFIDF:
        def __init__(self):
            raise AssertionError("auto should not try TF-IDF fallback")

    monkeypatch.setenv("MEMK_EMBEDDER", "auto")
    monkeypatch.setattr(embedder_mod, "_DEFAULT_EMBEDDER", None)
    monkeypatch.setattr(embedder_mod, "_DEFAULT_PIPELINE", None)
    monkeypatch.setattr(embedder_mod, "SentenceTransformerEmbedder", MissingSentence)
    monkeypatch.setattr(embedder_mod, "TFIDFEmbedder", UnexpectedTFIDF)

    embedder = embedder_mod.get_default_embedder()

    assert isinstance(embedder, embedder_mod.HashingEmbedder)
