from enum import Enum
from typing import Optional

from pydantic import field_validator

from Utils.YamlModel import YamlModel


class EmbeddingType(Enum):
    OPENAI = "openai"
    HF = "hf"
    OLLAMA = "ollama"
    MODELSCOPE = "modelscope"


class EmbeddingConfig(YamlModel):
    """Option for Embedding.
    """

    api_type: Optional[EmbeddingType] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None

    model: Optional[str] = None
    cache_folder: Optional[str] = None
    target_devices: Optional[list[str]] = None
    embed_batch_size: Optional[int] = None
    dimensions: Optional[int] = None  # output dimension of embedding model
    backend: Optional[str] = None
    @field_validator("api_type", mode="before")
    @classmethod
    def check_api_type(cls, v):
        if v == "":
            return None
        return v
