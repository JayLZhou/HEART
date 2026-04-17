from Common.Utils import mdhash_id
from Common.Logger import logger
import os
import json
import faiss
from typing import Any
from llama_index.core.schema import (
    Document,
    TextNode
)
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, Settings
from Index.BaseIndex import BaseIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import QueryBundle
import numpy as np
from llama_index.vector_stores.faiss import FaissVectorStore
from concurrent.futures import ProcessPoolExecutor
from Index.EmbeddingFactory import get_rag_embedding
from tqdm import tqdm

class FaissIndex(BaseIndex):
    """FaissIndex is designed to be simple and straightforward.

    It is a lightweight and easy-to-use vector database for ANN search.
    """

    def __init__(self, config):
        super().__init__(config)
        print(config)
        # self.embedding_model = get_rag_embedding(self.config.embedding.api_type, self.config)
        self.embedding_model = self.config.embed_model

    def _metric_constant(self):
        metric = str(getattr(self.config, "metric", "l2")).strip().lower()
        if metric == "l2":
            return faiss.METRIC_L2
        if metric == "inner_product":
            return faiss.METRIC_INNER_PRODUCT
        raise ValueError(f"Unsupported FAISS metric: {self.config.metric}")

    def _build_faiss_index(self):
        index = faiss.IndexHNSWFlat(
            int(self.config.dimensions),
            int(getattr(self.config, "hnsw_m", 32)),
            self._metric_constant(),
        )
        index.hnsw.efSearch = int(getattr(self.config, "hnsw_ef_search", 64))
        index.hnsw.efConstruction = int(getattr(self.config, "hnsw_ef_construction", 40))
        return index

    def _metadata_path(self) -> str:
        return os.path.join(str(self.config.persist_path), "faiss_meta.json")

    def _expected_metadata(self) -> dict[str, Any]:
        return {
            "dimensions": int(self.config.dimensions),
            "hnsw_m": int(getattr(self.config, "hnsw_m", 32)),
            "hnsw_ef_search": int(getattr(self.config, "hnsw_ef_search", 64)),
            "hnsw_ef_construction": int(getattr(self.config, "hnsw_ef_construction", 40)),
            "metric": str(getattr(self.config, "metric", "l2")).strip().lower(),
        }

    def _write_metadata(self):
        os.makedirs(str(self.config.persist_path), exist_ok=True)
        with open(self._metadata_path(), "w", encoding="utf-8") as f:
            json.dump(self._expected_metadata(), f, ensure_ascii=False, indent=2, sort_keys=True)

    def _metadata_matches(self) -> bool:
        path = self._metadata_path()
        if not os.path.exists(path):
            logger.warning("FAISS metadata file missing at %s; index will be rebuilt.", path)
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                current = json.load(f)
        except Exception as e:
            logger.warning("Failed to read FAISS metadata from %s: %s", path, e)
            return False
        expected = self._expected_metadata()
        if current != expected:
            logger.warning("FAISS metadata mismatch. expected=%s current=%s", expected, current)
            return False
        return True
    def retrieval(self, query, top_k):
        if top_k is None:
            top_k = self._get_retrieve_top_k()
        retriever = self._index.as_retriever(similarity_top_k=top_k, embed_model=self.embedding_model)
        query_emb = self._embed_text(query)
        query_bundle = QueryBundle(query_str=query, embedding=query_emb)
    
        # TODO: async
        # return retriever.aretrieve(query_bundle)
        return retriever.retrieve(query_bundle)

    def get_retriever(self, top_k):
        return self._index.as_retriever(similarity_top_k=top_k, embed_model=self.embedding_model)

    def retrieval_batch(self, queries, top_k):
        pass
    def _embed_text(self, text: str):
        return self.embedding_model._get_text_embedding(text)
    
    def _update_index(self, datas: list[dict[str:Any]], meta_data: list):
        def process_document(data):
     
            document = Document(
                doc_id=data[0],
                text=data[1].content,
                metadata={key: data[key] for key in meta_data},
                excluded_embed_metadata_keys=meta_data,
            )
            return document
        Settings.embed_model = self.embedding_model
        documents = [process_document(data) for data in datas]
        texts = [doc.text for doc in documents] 
      
        # Generate embeddings with progress bar
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        # Use batch processing with progress bar
        text_embeddings = self.embedding_model._get_text_embeddings(texts)

        vector_store = FaissVectorStore(faiss_index=self._build_faiss_index())
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self._index =  VectorStoreIndex([], storage_context=storage_context,
            embed_model= self.embedding_model)
      
      
        
        nodes = []
        for doc, embedding in zip(documents, text_embeddings):
            node = TextNode(text=doc.text, embedding=embedding, metadata=doc.metadata)
            nodes.append(node)
        self._index.insert_nodes(nodes)


          
        
        logger.info("refresh index size is {}".format(len(documents)))

    def _load_index(self) -> bool:
        try:
            if not self._metadata_matches():
                return False
            Settings.embed_model = self.embedding_model

            vector_store = FaissVectorStore.from_persist_dir(str(self.config.persist_path))
            vector_store.client.hnsw.efSearch = int(getattr(self.config, "hnsw_ef_search", 64))
            vector_store.client.hnsw.efConstruction = int(getattr(self.config, "hnsw_ef_construction", 40))
  
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=self.config.persist_path)
     
            self._index  =load_index_from_storage(storage_context=storage_context, embed_model=self.embedding_model)

            return True
        except Exception as e:
            logger.error("Loading index error: {}".format(e))
            return False

    def upsert(self, data: dict[str: Any]):
        pass

    def exist_index(self):
        return os.path.exists(self.config.persist_path)

    def _get_retrieve_top_k(self):
        return self.config.retrieve_top_k

    def _storage_index(self):
        self._index.storage_context.persist(persist_dir=self.config.persist_path)
        self._write_metadata()

    def _update_index_from_documents(self, docs: list[Document]):
        refreshed_docs = self._index.refresh_ref_docs(docs)

        # the number of docs that are refreshed. if True in refreshed_docs, it means the doc is refreshed.
        logger.info("refresh index size is {}".format(len([True for doc in refreshed_docs if doc])))

    def _get_index(self):
        Settings.embed_model = self.embedding_model
        vector_store = FaissVectorStore(faiss_index=self._build_faiss_index())
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return  VectorStoreIndex(
            nodes = [],
            storage_context=storage_context,
            embed_model= self.embedding_model,
        )   
   

    def _similarity_score(self, object_q, object_d):
        # For llama_index based vector database, we do not need it now!
        pass

   
