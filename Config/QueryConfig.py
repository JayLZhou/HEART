#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Optional, List
from pydantic import BaseModel


class QueryConfig(BaseModel):
    """Query configuration"""
    
    # Query processing
    enable_query_expansion: bool = False
    query_expansion_method: str = "llm"  # llm, keyword, embedding
    
    # Query rewriting
    enable_query_rewriting: bool = False
    query_rewrite_prompt_path: Optional[str] = None
    
    # Query decomposition
    enable_query_decomposition: bool = False
    max_subqueries: int = 3
    
    # Query type detection
    enable_query_type_detection: bool = False
    query_types: List[str] = ["factoid", "list", "comparison", "reasoning"]
    
    # Answer generation
    answer_prompt_path: Optional[str] = None
    enable_citation: bool = True
    max_answer_length: int = 512
    
    # Multi-hop reasoning
    enable_multi_hop: bool = False
    max_reasoning_steps: int = 3
    
    # Conversation
    enable_conversation_history: bool = False
    max_history_length: int = 5
    
    # Response format
    response_format: str = "text"  # text, json, markdown
    
    # Quality control
    enable_answer_verification: bool = False
    min_confidence_score: float = 0.6



