#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pydantic import BaseModel


class ChunkConfig(BaseModel):
    """Chunk configuration"""
    
    # Chunking method
    chunk_method: str = "chunking_by_token_size"  # chunking_by_token_size, semantic, recursive
    
    # Token-based chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    
    # Character-based chunking (as fallback)
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Semantic chunking
    semantic_similarity_threshold: float = 0.7
    semantic_min_chunk_size: int = 100
    semantic_max_chunk_size: int = 2000
    
    # Recursive chunking
    separators: list = ["\n\n", "\n", ". ", " ", ""]
    
    # Document processing
    enable_metadata_extraction: bool = True
    preserve_structure: bool = True  # Preserve headers, lists, etc.
    
    # Entity linking in chunks
    enable_entity_linking: bool = True
    entity_linking_threshold: float = 0.8
    
    # Chunk filtering
    min_chunk_length: int = 10
    max_chunk_length: int = 5000
    
    # Special handling
    handle_code_blocks: bool = True
    handle_tables: bool = True
    
    # Tokenizer
    tokenizer_model: str = "gpt-3.5-turbo"

