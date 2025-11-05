#!/usr/bin/env python
# -*- coding: utf-8 -*-
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class LLMType(Enum):
    OPENAI = "openai"
    FIREWORKS = "fireworks"
    OPEN_LLM = "open_llm"
    OLLAMA = "ollama"  # /chat at ollama api
    OLLAMA_GENERATE = "ollama.generate"  # /generate at ollama api
    OLLAMA_EMBEDDINGS = "ollama.embeddings"  # /embeddings at ollama api
    OLLAMA_EMBED = "ollama.embed"  # /embed at ollama api
    OPENROUTER = "openrouter"
    BEDROCK = "bedrock"
    ARK = "ark"  # https://www.volcengine.com/docs/82379/1263482#python-sdk


class LLMConfig(BaseModel):
    """LLM configuration"""
    
    # API type
    api_type: LLMType = LLMType.OPENAI
    
    # Model name
    model: str = "gpt-3.5-turbo"
    
    # API key
    api_key: Optional[str] = None
    
    # API base URL
    api_base: Optional[str] = None
    
    # API version (for Azure)
    api_version: Optional[str] = None
    
    # Organization ID (for OpenAI)
    organization: Optional[str] = None
    
    # Temperature
    temperature: float = 0.0
    
    # Max tokens
    max_tokens: int = 2048
    
    # Top p
    top_p: float = 1.0
    
    # Frequency penalty
    frequency_penalty: float = 0.0
    
    # Presence penalty
    presence_penalty: float = 0.0
    
    # Timeout
    timeout: int = 60
    
    # Max retries
    max_retries: int = 3
    
    # Model path for local models
    model_path: Optional[str] = None
    
    class Config:
        use_enum_values = True



