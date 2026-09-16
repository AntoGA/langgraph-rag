from fastapi.testclient import TestClient

from rag import api


def test_health_and_validation():
    client = TestClient(api.app)
    assert client.get("/health").status_code == 200
    assert client.post("/ask", json={"question": "   "}).status_code == 422


def test_response(monkeypatch):
    monkeypatch.setattr(api, "answer", lambda q: {
        "answer": "Example", "citations": [], "is_answered": False,
        "trace": ["fallback"], "docs": [{"text": "private context"}]
    })
    response = TestClient(api.app).post("/ask", json={"question": "test"})
    assert response.status_code == 200
    assert "docs" not in response.json()


def test_errors_are_not_exposed(monkeypatch):
    def fail(question):
        raise RuntimeError("secret provider details")
    monkeypatch.setattr(api, "answer", fail)
    response = TestClient(api.app).post("/ask", json={"question": "test"})
    assert response.status_code == 503
    assert "secret" not in response.text
