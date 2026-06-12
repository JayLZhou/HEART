from __future__ import annotations
import typing as T
from typing import Any
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    HuggingFaceEmbedding = None
    print("HuggingFaceEmbedding not found, please install it by `pip install huggingface_hub`")

from Index.TextEmbedding import TextEmbedding



from Config.EmbConfig import EmbeddingType
from Config.LLMConfig import LLMType
from Common.BaseFactory import GenericFactory
from Common.Logger import logger
from Option.Config2 import Config



class RAGEmbeddingFactory(GenericFactory):
    """Unified factory for creating embedding models.
    
    Supports:
    - LlamaIndex embeddings: OpenAI, Ollama, HuggingFace
    - Custom TextEmbeddingProvider: ModelScope local/ollama/openai
    """

    def __init__(self):
        creators = {
            EmbeddingType.OPENAI: self._create_openai,
            EmbeddingType.OLLAMA: self._create_ollama,
            EmbeddingType.HF: self._create_hf,
            EmbeddingType.MODELSCOPE: self._create_modelscope,  # Custom backend
        }
        super().__init__(creators)

    def get_rag_embedding(
        self, key: EmbeddingType = None, config: Config = None
    ) -> BaseEmbedding:
        """Get embedding instance by key or config."""
        return super().get_instance(key or self._resolve_embedding_type(config), config=config)

    @staticmethod
    def _resolve_embedding_type(config) -> EmbeddingType | LLMType:
        """Resolve the embedding type from config."""
        if config.embedding.api_type:
            return config.embedding.api_type
        raise TypeError("To use RAG, please set your embedding in Config2.yaml.")

    def _create_openai(self, config) -> OpenAIEmbedding:
        """Create OpenAI embedding."""
        params = dict(
            api_key=config.embedding.api_key or config.llm.api_key,
            api_base=config.embedding.base_url or config.llm.base_url,
        )
        self._try_set_model_and_batch_size(params, config)
        return OpenAIEmbedding(**params)

    def _create_ollama(self, config):
        """Create Ollama embedding."""
        from llama_index.embeddings.ollama import OllamaEmbedding

        params = dict(base_url=config.embedding.base_url)
        self._try_set_model_and_batch_size(params, config)
        return OllamaEmbedding(**params)

    def _create_hf(self, config) -> HuggingFaceEmbedding:
        """Create HuggingFace embedding."""
        if HuggingFaceEmbedding is None:
            raise ImportError("HuggingFaceEmbedding is not available.")
        
        params = dict(
            model_name=config.embedding.model,
            cache_folder=config.embedding.cache_folder,
            device="cuda",
            target_devices=config.embedding.target_devices,
            embed_batch_size=config.embedding.embed_batch_size,
        )
        
        # Set device: prefer target_devices if specified, otherwise auto-detect
        if config.embedding.target_devices:
            if isinstance(config.embedding.target_devices, list) and len(config.embedding.target_devices) > 0:
                params["device"] = config.embedding.target_devices[0]
                params["target_devices"] = config.embedding.target_devices
            elif isinstance(config.embedding.target_devices, str):
                params["device"] = config.embedding.target_devices
     
        if config.embedding.cache_folder == "":
            del params["cache_folder"]
        return HuggingFaceEmbedding(**params)

    def _create_modelscope(self, config) -> TextEmbedding:
        """Create ModelScope embedding provider with adapter."""
        # Determine backend from config
        backend = getattr(config.embedding, "backend", "local")
        
        # Handle device parameter: use target_devices if available, otherwise "auto"
        device = config.embedding.target_devices if config.embedding.target_devices else "auto"
        
        provider = TextEmbedding(
            model_name=config.embedding.model or "Qwen/Qwen3-Embedding-4B",
            backend=backend,
            device=device,
            max_length=getattr(config.embedding, "max_length", 8192),
            api_base=config.embedding.base_url,
            batch_size=config.embedding.embed_batch_size or 16,
        )
        
        return provider
    @staticmethod
    def _try_set_model_and_batch_size(params: dict, config):
        """Set model_name and embed_batch_size from config."""
        if config.embedding.model:
            params["model_name"] = config.embedding.model

        if config.embedding.embed_batch_size:
            params["embed_batch_size"] = config.embedding.embed_batch_size

        if config.embedding.dimensions:
            params["dimensions"] = config.embedding.dimensions

    def _raise_for_key(self, key: Any):
        raise ValueError(f"The embedding type is currently not supported: `{type(key)}`, {key}")



get_rag_embedding = RAGEmbeddingFactory().get_rag_embedding
