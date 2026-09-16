from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: SecretStr = SecretStr("")
    generation_model: str = ""
    utility_model: str = ""
    data_dir: Path = Path("data/raw")
    index_path: Path = Path("data/index/chunks.json")
    retrieval_mode: Literal["bm25", "hybrid"] = "bm25"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    dense_weight: float = Field(0.6, ge=0, le=1)
    top_k: int = Field(4, ge=1, le=20)
    chunk_size: int = Field(900, ge=50)
    chunk_overlap: int = Field(150, ge=0)
    max_retrieval_attempts: int = Field(2, ge=1, le=5)
    max_generation_attempts: int = Field(2, ge=1, le=5)

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self
