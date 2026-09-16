import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .runtime import answer

app = FastAPI(title="LangGraph RAG", version="0.1.0")
logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value.strip()


class AskResponse(BaseModel):
    answer: str
    citations: list[dict]
    is_answered: bool
    trace: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        result = answer(request.question)
    except Exception as exc:
        logger.exception("Pipeline failed")
        raise HTTPException(503, "Pipeline unavailable; check server configuration and logs") from exc
    return {key: result[key] for key in ("answer", "citations", "is_answered", "trace")}
