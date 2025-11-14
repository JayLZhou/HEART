# vector_index_bm25s.py

import os
import asyncio
import numpy as np
import pickle
from typing import Any, List
from nltk.tokenize import word_tokenize
import bm25s
from Common.Utils import mdhash_id
from Common.Logger import logger
from llama_index.core.schema import Document
from Index.BaseIndex import BaseIndex, VectorIndexNodeResult
from llama_index.core.schema import MetadataMode
from llama_index.core.node_parser import SimpleNodeParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import Stemmer

class BMIndex(BaseIndex):
    """VectorIndex using bm25s implementation."""

    def __init__(self, config):
        super().__init__(config)
        self._bm25_index = None
        self.max_workers = 16
        self._raw_documents = []  # original texts
        self._nodes = []  # store nodes for retrieval
        self._docid_to_metadata = {}  # optional metadata map

    async def retrieval(self, query, top_k):
        if top_k is None:
            top_k = self._get_retrieve_top_k()
        
        if self._bm25_index is None:
            logger.warning("BM25 index is not initialized")
            return []
        
        tokens = bm25s.tokenize(query.lower(), stopwords="en", stemmer=Stemmer.Stemmer("english"))
        results, scores = self._bm25_index.retrieve(tokens, k=top_k)
        
        retrieval_results = []
        for idx, score in zip(results[0], scores[0]):
            if idx < len(self._nodes):
                node = self._nodes[idx]
                retrieval_results.append({
                    "text": node.get_content(metadata_mode=MetadataMode.EMBED),
                    "score": float(score),
                    "metadata": node.metadata
                })
        return retrieval_results

    async def retrieval_nodes(self, query, top_k, graph, need_score=False, tree_node=False):
        results = await self.retrieval(query, top_k)
        result = VectorIndexNodeResult(results)
        if tree_node:
            return await result.get_tree_node_data(graph, need_score)
        else:
            return await result.get_node_data(graph, need_score)



    async def retrieval_nodes_with_score_matrix(self, query_list, top_k, graph):
        if isinstance(query_list, str):
            query_list = [query_list]
        results = await asyncio.gather(
            *[self.retrieval_nodes(query, top_k, graph, need_score=True) for query in query_list])
        reset_prob_matrix = np.zeros((len(query_list), graph.node_num))
        entity_indices = []
        scores = []

        async def set_idx_score(res):
            for entity, score in zip(res[0], res[1]):
                entity_indices.append(await graph.get_node_index(entity["entity_name"]))
                scores.append(score)

        await asyncio.gather(*[set_idx_score(res) for res in results])
        reset_prob_matrix[np.arange(len(query_list)).reshape(-1, 1), entity_indices] = scores
        all_entity_weights = reset_prob_matrix.max(axis=0)
        if all_entity_weights.sum() > 0:
            all_entity_weights /= all_entity_weights.sum()
        return all_entity_weights

    def _update_index(self, datas: List[dict[str, Any]], meta_data: List[str]):
        def process_document(data):
            document = Document(
                doc_id=data[0],
                text=data[1].content,
                metadata={key: data[key] for key in meta_data},
                excluded_embed_metadata_keys=meta_data,
            )
            return document
        
        completed_list = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for j in range(0, self.max_workers):
                process_tasks = [pool.submit(process_document,
                                         data=datas[i]) for i in range(len(datas)) if i % self.max_workers == j]
                completed_list.extend(as_completed(process_tasks))
        
        documents = []                
        for task in completed_list:
            documents.append(task.result())
        
        parser = SimpleNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents(documents)
        
        # Store nodes for later retrieval
        self._nodes = nodes
        
        # Store raw documents and metadata
        self._raw_documents = [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes]
        for node in nodes:
            doc_id = node.node_id
            self._docid_to_metadata[doc_id] = node.metadata

        k1 = getattr(self.config, "k1", 1.5)
        b = getattr(self.config, "b", 0.75)
        self._bm25_index = bm25s.BM25(k1=k1, b=b)

        corpus_tokens = bm25s.tokenize(
                self._raw_documents,
                stopwords="en",
                stemmer=Stemmer.Stemmer("english"),
                show_progress=False,
            )
        self._bm25_index.index(corpus_tokens, show_progress=False)

        logger.info(f"BM25 index built with k1={k1}, b={b}, size={len(documents)}.")

    def _load_index(self) -> bool:
        try:
            index_path = os.path.join(self.config.persist_path, "bm25_index")
            nodes_path = os.path.join(self.config.persist_path, "nodes.pkl")
            metadata_path = os.path.join(self.config.persist_path, "metadata.pkl")
            docs_path = os.path.join(self.config.persist_path, "raw_docs.pkl")
            
            if not os.path.exists(index_path):
                logger.warning(f"BM25 index path does not exist: {index_path}")
                return False
            
            # Load BM25 index
            self._bm25_index = bm25s.BM25.load(index_path, mmap=False)
            
            # Load nodes
            if os.path.exists(nodes_path):
                with open(nodes_path, "rb") as f:
                    self._nodes = pickle.load(f)
            
            # Load metadata
            if os.path.exists(metadata_path):
                with open(metadata_path, "rb") as f:
                    self._docid_to_metadata = pickle.load(f)
            
            # Load raw documents
            if os.path.exists(docs_path):
                with open(docs_path, "rb") as f:
                    self._raw_documents = pickle.load(f)
            
            logger.info(f"BM25 index loaded from {self.config.persist_path}, size={len(self._nodes)}")
            return True
        except Exception as e:
            logger.error(f"BM25 load failed: {e}")
        return False

    def _get_retrieve_top_k(self):
        return self.config.retrieve_top_k

    def _storage_index(self):
        os.makedirs(self.config.persist_path, exist_ok=True)
        
        # Save BM25 index
        index_path = os.path.join(self.config.persist_path, "bm25_index")
        if self._bm25_index is not None:
            self._bm25_index.save(index_path)
        
        # Save nodes
        nodes_path = os.path.join(self.config.persist_path, "nodes.pkl")
        with open(nodes_path, "wb") as f:
            pickle.dump(self._nodes, f)
        
        # Save metadata
        metadata_path = os.path.join(self.config.persist_path, "metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(self._docid_to_metadata, f)
        
        # Save raw documents
        docs_path = os.path.join(self.config.persist_path, "raw_docs.pkl")
        with open(docs_path, "wb") as f:
            pickle.dump(self._raw_documents, f)
        
        logger.info(f"BM25 index saved to {self.config.persist_path}, size={len(self._nodes)}")

    async def upsert(self, data: dict[str, Any]):
        pass

    async def retrieval_batch(self, queries, top_k):
        pass

    def exist_index(self):
        index_path = os.path.join(self.config.persist_path, "bm25_index")
        return os.path.exists(index_path)

    def _get_index(self):
        return None

    async def _update_index_from_documents(self, docs: List[Document]):
        pass

    async def _similarity_score(self, object_q, object_d):
        pass
