#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Optional, List
from pydantic import BaseModel


class RetrieverConfig(BaseModel):
    """Retriever configuration"""
    
    # Retrieval method
    retrieval_method: str = "hybrid"  # vector, graph, hybrid
    
    # Vector retrieval
    top_k: int = 10
    similarity_threshold: float = 0.7
    
    # Graph retrieval
    graph_top_k: int = 5
    enable_graph_expansion: bool = True
    graph_expansion_depth: int = 1
    
    # Hybrid retrieval weights
    vector_weight: float = 0.5
    graph_weight: float = 0.5
    
    # Reranking
    enable_reranking: bool = False
    rerank_model: Optional[str] = None
    rerank_top_k: int = 5
    
    # Diversity
    enable_diversity: bool = False
    diversity_lambda: float = 0.5  # MMR lambda
    
    # Filtering
    enable_metadata_filtering: bool = False
    metadata_filters: Optional[dict] = None
    
    # Entity-based retrieval
    entity_retrieval_top_k: int = 10
    relation_retrieval_top_k: int = 5
    
    # Chunk retrieval
    enable_chunk_retrieval: bool = True
    chunk_retrieval_top_k: int = 5
    
    # Subgraph retrieval (for Medical-GraphRAG)
    enable_subgraph_retrieval: bool = False
    subgraph_retrieval_top_k: int = 3

