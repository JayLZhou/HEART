#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM Pool Manager - Manages multiple LLM configurations and instances
"""

from typing import Dict, List, Optional
from Config.LLMConfig import LLMConfig
from Provider.BaseLLM import BaseLLM
from Provider.LLMProviderRegister import create_llm_instance
from Common.CostManager import CostManager, FireworksCostManager, TokenCostManager
from Config.LLMConfig import LLMType


class LLMPool:
    """LLM Pool Manager - manages multiple LLM configurations and instances"""
    
    def __init__(self):
        self._llm_configs: Dict[str, LLMConfig] = {}
        self._llm_instances: Dict[str, BaseLLM] = {}
        self._default_model: Optional[str] = None
        
    def register_llm(self, llm_config: LLMConfig, is_default: bool = False) -> None:
        """Register a LLM configuration to the pool
        
        Args:
            llm_config: LLM configuration
            is_default: Whether this is the default LLM
        """
        model_name = llm_config.model
        if not model_name:
            raise ValueError("LLM config must have a model name")
            
        self._llm_configs[model_name] = llm_config
        
        if is_default or self._default_model is None:
            self._default_model = model_name
            
    def register_llms(self, llm_configs: List[LLMConfig], default_model: Optional[str] = None) -> None:
        """Register multiple LLM configurations to the pool
        
        Args:
            llm_configs: List of LLM configurations
            default_model: Name of the default model (if None, use the first one)
        """
        if not llm_configs:
            raise ValueError("Must provide at least one LLM configuration")
            
        for idx, llm_config in enumerate(llm_configs):
            is_default = (idx == 0 and default_model is None) or (llm_config.model == default_model)
            self.register_llm(llm_config, is_default=is_default)
            
    def get_llm(self, model_name: Optional[str] = None, cost_manager: Optional[CostManager] = None) -> BaseLLM:
        """Get a LLM instance by model name
        
        Args:
            model_name: Model name (if None, use default model)
            cost_manager: Cost manager instance (if None, create a new one)
            
        Returns:
            BaseLLM instance
        """
        # Use default model if no model name provided
        if model_name is None:
            model_name = self._default_model
            
        if model_name is None:
            raise ValueError("No default model set and no model name provided")
            
        if model_name not in self._llm_configs:
            raise ValueError(f"Model '{model_name}' not found in LLM pool. Available models: {list(self._llm_configs.keys())}")
        
        # Return cached instance if available
        if model_name in self._llm_instances:
            return self._llm_instances[model_name]
            
        # Create new LLM instance
        llm_config = self._llm_configs[model_name]
        llm = create_llm_instance(llm_config)
        
        # Set cost manager
        if llm.cost_manager is None:
            if cost_manager is None:
                cost_manager = self._select_costmanager(llm_config)
            llm.cost_manager = cost_manager
            
        # Cache the instance
        self._llm_instances[model_name] = llm
        
        return llm
        
    def _select_costmanager(self, llm_config: LLMConfig) -> CostManager:
        """Select appropriate cost manager based on LLM config
        
        Args:
            llm_config: LLM configuration
            
        Returns:
            CostManager instance
        """
        if llm_config.api_type == LLMType.FIREWORKS:
            return FireworksCostManager()
        elif llm_config.api_type == LLMType.OPEN_LLM:
            return TokenCostManager()
        else:
            return CostManager()
            
    def get_llm_config(self, model_name: str) -> LLMConfig:
        """Get LLM configuration by model name
        
        Args:
            model_name: Model name
            
        Returns:
            LLMConfig instance
        """
        if model_name not in self._llm_configs:
            raise ValueError(f"Model '{model_name}' not found in LLM pool")
        return self._llm_configs[model_name]
        
    def has_model(self, model_name: str) -> bool:
        """Check if a model exists in the pool
        
        Args:
            model_name: Model name
            
        Returns:
            True if model exists, False otherwise
        """
        return model_name in self._llm_configs
        
    def list_models(self) -> List[str]:
        """Get list of all registered model names
        
        Returns:
            List of model names
        """
        return list(self._llm_configs.keys())
        
    def get_default_model(self) -> Optional[str]:
        """Get the default model name
        
        Returns:
            Default model name
        """
        return self._default_model
        
    def clear_cache(self) -> None:
        """Clear all cached LLM instances"""
        self._llm_instances.clear()


# Global LLM pool instance
_global_llm_pool: Optional[LLMPool] = None


def get_global_llm_pool() -> LLMPool:
    """Get the global LLM pool instance
    
    Returns:
        Global LLMPool instance
    """
    global _global_llm_pool
    if _global_llm_pool is None:
        _global_llm_pool = LLMPool()
    return _global_llm_pool


def reset_global_llm_pool() -> None:
    """Reset the global LLM pool instance"""
    global _global_llm_pool
    _global_llm_pool = None




