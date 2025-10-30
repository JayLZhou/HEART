#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Optional
from pydantic import BaseModel


class GraphConfig(BaseModel):
    """Graph configuration"""
    
    # Graph storage type
    graph_storage_type: str = "networkx"  # networkx, neo4j, etc.
    
    # Graph database connection (for Neo4j)
    graph_db_uri: Optional[str] = None
    graph_db_user: Optional[str] = None
    graph_db_password: Optional[str] = None
    
    # Entity extraction
    entity_extract_max_gleaning: int = 1
    entity_extract_prompt_path: Optional[str] = None
    
    # Relationship extraction
    relation_extract_prompt_path: Optional[str] = None
    
    # Community detection
    enable_community_detection: bool = True
    community_algorithm: str = "leiden"  # leiden, louvain, etc.
    community_resolution: float = 1.0
    
    # Graph construction
    max_entity_length: int = 128
    max_relation_length: int = 256
    
    # Entity linking threshold
    entity_similarity_threshold: float = 0.8
    
    # Graph pruning
    enable_graph_pruning: bool = False
    min_entity_occurrence: int = 1
    min_relation_occurrence: int = 1
    
    # Subgraph extraction
    subgraph_max_depth: int = 2
    subgraph_max_nodes: int = 50
    
    # Graph persistence
    graph_save_path: Optional[str] = None
    enable_graph_caching: bool = True



