#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM Pool Helper Functions - Convenient functions for using the LLM pool
"""

from typing import Optional, List
from Provider.BaseLLM import BaseLLM
from Provider.LLMPool import get_global_llm_pool


def get_llm_from_pool(model_name: Optional[str] = None) -> BaseLLM:
    """Get a LLM instance from the global pool
    
    Args:
        model_name: Model name (if None, use default model)
        
    Returns:
        BaseLLM instance
        
    Example:
        # Get default LLM
        llm = get_llm_from_pool()
        
        # Get specific LLM by model name
        llm = get_llm_from_pool("gpt-4o")
    """
    llm_pool = get_global_llm_pool()
    return llm_pool.get_llm(model_name)


def list_available_models() -> List[str]:
    """List all available model names in the pool
    
    Returns:
        List of model names
        
    Example:
        models = list_available_models()
        print(f"Available models: {models}")
    """
    llm_pool = get_global_llm_pool()
    return llm_pool.list_models()


def get_default_model_name() -> Optional[str]:
    """Get the default model name
    
    Returns:
        Default model name
        
    Example:
        default = get_default_model_name()
        print(f"Default model: {default}")
    """
    llm_pool = get_global_llm_pool()
    return llm_pool.get_default_model()


def has_model_in_pool(model_name: str) -> bool:
    """Check if a model exists in the pool
    
    Args:
        model_name: Model name to check
        
    Returns:
        True if model exists, False otherwise
        
    Example:
        if has_model_in_pool("gpt-4o"):
            llm = get_llm_from_pool("gpt-4o")
    """
    llm_pool = get_global_llm_pool()
    return llm_pool.has_model(model_name)



