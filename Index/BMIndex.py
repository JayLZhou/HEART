# vector_index_bm25s.py

import os
import asyncio
import numpy as np
from typing import Any, List
from nltk.tokenize import word_tokenize
import bm25s
from Core.Common.Utils import mdhash_id
from Core.Common.Logger import logger
from llama_index.core.schema import Document
from Core.Index.BaseIndex import BaseIndex, VectorIndexNodeResult, VectorIndexEdgeResult
from llama_index.core.schema import QueryBundle, MetadataMode
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import (
    BaseNode,
    IndexNode,
    NodeWithScore,
    QueryBundle,
    
)
from concurrent.futures import ThreadPoolExecutor, as_completed

from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer
class BMIndex(BaseIndex):
    """VectorIndex using bm25s implementation."""

    def __init__(self, config):
        super().__init__(config)
        self._bm25_index = None
        self.max_workers = 16
        self._raw_documents = []  # original texts
        self._tokenized_corpus = []  # tokenized texts
        self._docid_to_metadata = {}  # optional metadata map

    async def retrieval(self, query, top_k):
        if top_k is None:
            top_k = self._get_retrieve_top_k()
        tokens = bm25s.tokenize(query.lower())
        scores = self._bm25_index.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            doc_id = mdhash_id(self._raw_documents[idx])
            metadata = self._docid_to_metadata.get(doc_id, {})
            results.append({
                "text": self._raw_documents[idx],
                "score": scores[idx],
                "metadata": metadata
            })
        return results

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
        def tokenize(text):
            return word_tokenize(text.lower())

        def process_document(data):
            document = Document(
                doc_id=mdhash_id(data.text),
                text=data.text,
                metadata={"chunk_id": data.chunk_id},
                excluded_embed_metadata_keys=["chunk_id"],
            )
            return document
        completed_list = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for j in range(0, self.max_workers):
                process_tasks = [pool.submit(process_document,
                                         data = datas[i]) for i in range(len(datas)) if i % self.max_workers == j]
                completed_list.extend(as_completed(process_tasks))
        documents = []                
        for task in completed_list:
            documents.append(task.result())
        parser = SimpleNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents(documents)

        k1 = getattr(self.config, "k1", 1.5)
        b = getattr(self.config, "b", 0.75)
        self.bm25 = bm25s.BM25(k1=k1, b=b)

        corpus_tokens = bm25s.tokenize(
                [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes],
                stopwords="en",
                stemmer= Stemmer.Stemmer("english"),
                show_progress=False,
            )
        self.bm25.index(corpus_tokens, show_progress=False )

        logger.info(f"BM25 index built with k1={k1}, b={b}, size={len(self._raw_documents)}.")

    def _load_index(self) -> bool:
        try:
            path = os.path.join(self.config.persist_path, "bm25_docs.txt")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    self._raw_documents = [line.strip() for line in f.readlines()]
                self._tokenized_corpus = [word_tokenize(text.lower()) for text in self._raw_documents]
                k1 = getattr(self.config, "k1", 1.5)
                b = getattr(self.config, "b", 0.75)
                self._bm25_index = bm25s.BM25(self._tokenized_corpus, k1=k1, b=b)
                logger.info("BM25 index loaded from disk.")
                return True
        except Exception as e:
            logger.error(f"BM25 load failed: {e}")
        return False

    def _get_retrieve_top_k(self):
        return self.config.retrieve_top_k

    def _storage_index(self):
        os.makedirs(self.config.persist_path, exist_ok=True)
        with open(os.path.join(self.config.persist_path, "bm25_docs.txt"), "w", encoding="utf-8") as f:
            for text in self._raw_documents:
                f.write(text.strip() + "\n")

    async def upsert(self, data: dict[str, Any]):
        pass

    async def retrieval_batch(self, queries, top_k):
        pass

    def exist_index(self):
        return os.path.exists(self.config.persist_path)

    def _get_index(self):
        return None

    async def _update_index_from_documents(self, docs: List[Document]):
        pass

    async def _similarity_score(self, object_q, object_d):
        pass
