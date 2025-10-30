#!/usr/bin/env python
# -*- coding: utf-8 -*-
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class EmbeddingType(str, Enum):
    """Embedding type enumeration"""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OLLAMA = "ollama"


class EmbeddingConfig(BaseModel):
    """Embedding configuration"""
    
    # Embedding provider type
    embedding_type: EmbeddingType = EmbeddingType.OPENAI
    
    # Model name
    embedding_model: str = "text-embedding-ada-002"
    
    # API configuration for OpenAI
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # Model path for local models (HuggingFace, Sentence Transformers)
    model_path: Optional[str] = None
    
    # Embedding dimension
    embedding_dim: int = 1536
    
    # Batch size for embedding
    batch_size: int = 32
    
    # Device for local models
    device: str = "cuda"
    
    # Max tokens for embedding
    max_tokens: int = 8191
    
    class Config:
        use_enum_values = True



