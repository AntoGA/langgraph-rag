import hashlib
import json
import os
import re
import tempfile
from typing import TypedDict

import numpy as np
from rank_bm25 import BM25Okapi

from .config import Settings


class Chunk(TypedDict):
    id: str
    source: str
    text: str


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def ingest(settings: Settings) -> int:
    chunks: list[Chunk] = []
    for path in sorted(settings.data_dir.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        source = path.relative_to(settings.data_dir).as_posix()
        step = settings.chunk_size - settings.chunk_overlap
        for start in range(0, len(text), step):
            part = text[start:start + settings.chunk_size].strip()
            if not tokenize(part):
                continue
            digest = hashlib.sha256(f"{source}\0{start}\0{part}".encode()).hexdigest()
            chunks.append({"id": digest, "source": source, "text": part})
    if not chunks:
        raise ValueError("No nonempty MD/TXT documents; previous index left unchanged")
    target = settings.index_path
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         delete=False) as handle:
            temporary = handle.name
            json.dump(chunks, handle, ensure_ascii=False)
        os.replace(temporary, target)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return len(chunks)


def load_chunks(path) -> list[Chunk]:
    chunks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Invalid or empty index. Run rag ingest")
    for chunk in chunks:
        if not isinstance(chunk, dict) or not all(
            isinstance(chunk.get(key), str) and chunk[key] for key in ("id", "source", "text")
        ):
            raise ValueError("Invalid chunk")
    return chunks


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], settings: Settings):
        self.chunks = chunks
        self.settings = settings
        self.tokens = [tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokens)
        self.encoder = None
        if settings.retrieval_mode == "hybrid":
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(settings.embedding_model)
            self.vectors = self.encoder.encode(
                [chunk["text"] for chunk in chunks], normalize_embeddings=True
            )

    def __call__(self, query: str) -> list[Chunk]:
        terms = tokenize(query)
        if not terms:
            return []
        scores = self.bm25.get_scores(terms)
        # Exclude no-overlap candidates even if BM25 gives a tied zero score.
        sparse = [int(i) for i in np.argsort(-scores, kind="stable")
                  if set(terms).intersection(self.tokens[int(i)])]
        if self.encoder is None:
            return [self.chunks[i] for i in sparse[:self.settings.top_k]]
        vector = self.encoder.encode([query], normalize_embeddings=True)[0]
        dense = np.argsort(-(self.vectors @ vector), kind="stable")
        fused: dict[int, float] = {}
        for ranking, weight in ((sparse, 1 - self.settings.dense_weight),
                                (dense, self.settings.dense_weight)):
            if weight == 0:
                continue
            for rank, index in enumerate(ranking, 1):
                i = int(index)
                fused[i] = fused.get(i, 0) + weight / (60 + rank)
        ordered = sorted(fused, key=lambda i: (-fused[i], i))
        return [self.chunks[i] for i in ordered[:self.settings.top_k]]
