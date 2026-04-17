"""RAG schemas."""
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from llama_index.core.embeddings import BaseEmbedding
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

class BaseIndexConfig(BaseModel):
    """Common config for index.

    If add new subconfig, it is necessary to add the corresponding instance implementation in rag.factories.index.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_config['protected_namespaces'] = ()
    persist_path: Union[str, Path] = Field(description="The directory of saved data.")


class VectorIndexConfig(BaseIndexConfig):
    """Option for vector-based index."""

    embed_model: BaseEmbedding = Field(default=None, description="Embed model.")


class ColBertIndexConfig(BaseIndexConfig):
    """Option for colbert-based index."""
    index_name: str = Field(default="", description="The name of the index.")
    model_name: str = Field(default="colbert-ir/colbertv2.0", description="The name of the ColBERT model.")
    nbits: int = Field(default=2, description="Number of bits for quantization.")
    gpus: int = Field(default=0, description="Number of GPUs to use.")
    ranks: int = Field(default=1, description="Number of ranks for distributed indexing.")
    doc_maxlen: int = Field(default=120, description="Maximum length of documents.")
    query_maxlen: int = Field(default=60, description="Maximum length of queries.")
    kmeans_niters: int = Field(default=4, description="Number of iterations for K-means clustering.")



class FAISSIndexConfig(VectorIndexConfig):
    """Config for faiss-based index."""
    dimensions: int = Field(default=128, description="Dimensions of the embedding model.")
    hnsw_m: int = Field(default=32, description="HNSW graph out-degree M.")
    hnsw_ef_search: int = Field(default=64, description="HNSW efSearch for query-time search.")
    hnsw_ef_construction: int = Field(default=40, description="HNSW efConstruction for build-time graph construction.")
    metric: str = Field(default="l2", description="FAISS metric name: l2 or inner_product.")

class BMIndexConfig(VectorIndexConfig):
    """Config for BM index."""
    k1: float = 1.2
    b: float = 0.5    
