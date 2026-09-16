from rag.config import Settings
from rag.index import HybridRetriever, ingest, load_chunks


def test_idempotent_snapshot(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.md").write_text("retrieval search attempts", encoding="utf-8")
    (raw / "b.txt").write_text("holiday policy", encoding="utf-8")
    config = Settings(_env_file=None, data_dir=raw, index_path=tmp_path / "index.json")
    assert ingest(config) == 2
    first = load_chunks(config.index_path)
    assert ingest(config) == 2
    assert first == load_chunks(config.index_path)
    (raw / "b.txt").unlink()
    assert ingest(config) == 1
    assert load_chunks(config.index_path)[0]["id"] == first[0]["id"]


def test_sparse_search():
    config = Settings(_env_file=None, retrieval_mode="bm25")
    docs = [{"id": "a", "source": "a.md", "text": "retrieval attempts"},
            {"id": "b", "source": "b.md", "text": "holiday policy"}]
    search = HybridRetriever(docs, config)
    assert search("retrieval") == [docs[0]]
    assert search("unknown") == []
    assert search("") == []
