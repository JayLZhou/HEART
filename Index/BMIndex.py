# vector_index_bm25.py

import os
import asyncio
import numpy as np
from typing import Any, List
from Common.Utils import mdhash_id
from Common.Logger import logger
from llama_index.core.schema import Document
from Index.BaseIndex import BaseIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import StorageContext
from llama_index.core.schema import QueryBundle
import Stemmer


class BMIndex(BaseIndex):
    """VectorIndex using llama_index BM25Retriever implementation."""

    def __init__(self, config):
        super().__init__(config)
        self._retriever = None
        self._docstore = None
        self._nodes = []

    async def retrieval(self, query, top_k):
        if top_k is None:
            top_k = self._get_retrieve_top_k()
        
        if self._retriever is None:
            logger.warning("BM25 retriever is not initialized")
            return []
        
        # Create query bundle
        query_bundle = QueryBundle(query_str=query)
        
        # Retrieve nodes
        retrieved_nodes = await self._retriever.aretrieve(query_bundle)
        
        # Format results
        retrieval_results = []
        for node_with_score in retrieved_nodes[:top_k]:
            retrieval_results.append({
                "text": node_with_score.node.get_content(),
                "score": node_with_score.score if node_with_score.score is not None else 0.0,
                "metadata": node_with_score.node.metadata
            })
        
        return retrieval_results

    async def retrieval_batch(self, queries, top_k):
        """Batch retrieval for multiple queries."""
        if isinstance(queries, str):
            queries = [queries]
        
        results = await asyncio.gather(
            *[self.retrieval(query, top_k) for query in queries]
        )
        return results

    
    def get_retriever(self, top_k):
        return BM25Retriever.from_defaults(
            nodes=self._nodes,
            similarity_top_k=top_k,
            stemmer=Stemmer.Stemmer("english"),
            language="english",
        )


    def _update_index(self, datas: List[dict[str, Any]], meta_data: List[str]):
        """Build BM25 index from data."""
        def process_document(data):
            document = Document(
                doc_id=data[0],
                text=data[1].content,
                metadata={key: data[key] for key in meta_data},
                excluded_embed_metadata_keys=meta_data,
            )
            return document
        
        # Process documents
        documents = [process_document(data) for data in datas]
        
        # Parse documents into nodes
        parser = SimpleNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents(documents)
        self._nodes = nodes
        
        # Create docstore
        self._docstore = SimpleDocumentStore()
        self._docstore.add_documents(nodes)
        
        # Get BM25 parameters
        k1 = getattr(self.config, "k1", 1.5)
        b = getattr(self.config, "b", 0.75)
        

        
        logger.info(f"BM25 index built with k1={k1}, b={b}, size={len(documents)}.")

    def _load_index(self) -> bool:
        """Load BM25 index from disk."""
        try:
            persist_path = self.config.persist_path
            docstore_path = os.path.join(persist_path, "docstore.json")
            
            if not os.path.exists(docstore_path):
                logger.warning(f"BM25 docstore does not exist: {docstore_path}")
                return False
            
            # Load storage context
            storage_context = StorageContext.from_defaults(
                persist_dir=persist_path
            )
            
            self._docstore = storage_context.docstore
            
            # Get nodes from docstore
            nodes = list(self._docstore.docs.values())
            self._nodes = nodes
            
            if len(nodes) == 0:
                logger.warning("No nodes found in docstore")
                return False
            
            
            logger.info(f"BM25 index loaded from {persist_path}, size={len(nodes)}")
            return True
            
        except Exception as e:
            logger.error(f"BM25 load failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
        return False

    def _get_retrieve_top_k(self):
        """Get retrieval top-k parameter."""
        return self.config.retrieve_top_k

    def _storage_index(self):
        """Save BM25 index to disk."""
        try:
            os.makedirs(self.config.persist_path, exist_ok=True)
            
            if self._docstore is None:
                logger.warning("Docstore is None, cannot save")
                return
            
            # Create storage context and persist
            storage_context = StorageContext.from_defaults(docstore=self._docstore)
            storage_context.persist(persist_dir=self.config.persist_path)
            
            logger.info(f"BM25 index saved to {self.config.persist_path}, size={len(self._nodes)}")
            
        except Exception as e:
            logger.error(f"BM25 save failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def exist_index(self):
        """Check if index exists on disk."""
        docstore_path = os.path.join(self.config.persist_path, "docstore.json")
        return os.path.exists(docstore_path)

    def _get_index(self):
        """Get index object (not used for BM25)."""
        return None

    async def upsert(self, data: dict[str, Any]):
        """Upsert a document (not implemented yet)."""
        pass

    async def _update_index_from_documents(self, docs: List[Document]):
        """Update index from documents (not implemented yet)."""
        pass

    async def _similarity_score(self, object_q, object_d):
        """Calculate similarity score (not needed for BM25)."""
        pass
