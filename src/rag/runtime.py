from functools import lru_cache
from threading import Lock

from .config import Settings
from .index import HybridRetriever, load_chunks
from .models import AnthropicModels
from .pipeline import build_graph, run

_LOCK = Lock()


@lru_cache(maxsize=1)
def _runtime():
    settings = Settings()
    chunks = load_chunks(settings.index_path)
    models = AnthropicModels(settings)
    return build_graph(models, HybridRetriever(chunks, settings), settings), settings


def answer(question):
    with _LOCK:
        graph, settings = _runtime()
    return run(graph, question, settings)
