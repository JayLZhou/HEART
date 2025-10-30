#!/usr/bin/env python
# -*- coding: utf-8 -*-
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class LLMType(str, Enum):
    """LLM type enumeration"""
    OPENAI = "openai"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    VLLM = "vllm"


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



