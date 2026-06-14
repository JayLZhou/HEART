from Common.Utils import mdhash_id
from Common.Logger import logger
import os
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

    def _metric_type(self) -> int:
        metric_name = getattr(self.config, "metric", "l2")
        if metric_name == "inner_product":
            return faiss.METRIC_INNER_PRODUCT
        return faiss.METRIC_L2

    def _apply_hnsw_runtime_params(self, faiss_index) -> None:
        hnsw = getattr(faiss_index, "hnsw", None)
        if hnsw is None:
            return
        hnsw.efSearch = int(getattr(self.config, "hnsw_ef_search", 64))
        hnsw.efConstruction = int(getattr(self.config, "hnsw_ef_construction", 40))

    def _build_faiss_index(self, dimensions: int):
        faiss_index = faiss.IndexHNSWFlat(
            dimensions,
            int(getattr(self.config, "hnsw_m", 32)),
            self._metric_type(),
        )
        self._apply_hnsw_runtime_params(faiss_index)
        return faiss_index

    def _get_vector_store(self, dimensions: int):
        return FaissVectorStore(faiss_index=self._build_faiss_index(dimensions))

    def _ensure_runtime_search_params(self) -> None:
        vector_store = getattr(getattr(self, "_index", None), "vector_store", None)
        if vector_store is None:
            storage_context = getattr(self._index, "storage_context", None)
            vector_store = getattr(storage_context, "vector_store", None)
        faiss_index = getattr(vector_store, "_faiss_index", None) or getattr(vector_store, "faiss_index", None)
        if faiss_index is not None:
            self._apply_hnsw_runtime_params(faiss_index)

    def retrieval(self, query, top_k):
        if top_k is None:
            top_k = self._get_retrieve_top_k()
        self._ensure_runtime_search_params()
        retriever = self._index.as_retriever(similarity_top_k=top_k, embed_model=self.embedding_model)
        query_emb = self._embed_text(query)
        query_bundle = QueryBundle(query_str=query, embedding=query_emb)
    
        # TODO: async
        # return retriever.aretrieve(query_bundle)
        return retriever.retrieve(query_bundle)

    def get_retriever(self, top_k):
        self._ensure_runtime_search_params()
        return self._index.as_retriever(similarity_top_k=top_k, embed_model=self.embedding_model)

    def retrieval_batch(self, queries, top_k):
        pass
    def _embed_text(self, text: str):
        return self.embedding_model._get_text_embedding(text)
    
    def _embed_texts_cached(self, texts: list) -> list:
        """Embed chunk texts, reusing a content-addressed on-disk cache.

        Chunk embeddings depend only on (chunk text, embedding model, dims) -- NOT on
        the FAISS/HNSW index params. Without this cache, force_rebuild=True re-embeds
        the whole corpus on every index rebuild (i.e. every tuning trial whose FAISS
        params differ), which dominates run time. The key hashes the model, dims and
        all chunk texts, so a cache hit always returns the correct vectors. Override the
        cache location with the HEART_EMB_CACHE env var.
        """
        import hashlib
        model_name = getattr(self.embedding_model, "model_name", None) or type(self.embedding_model).__name__
        dims = int(getattr(self.config, "dimensions", 0) or 0)
        h = hashlib.md5()
        h.update(f"{model_name}|{dims}|{len(texts)}|".encode("utf-8"))
        for t in texts:
            h.update(b"\x00")
            h.update((t or "").encode("utf-8", "ignore"))
        key = h.hexdigest()

        cache_dir = os.environ.get("HEART_EMB_CACHE") or os.path.join(
            os.path.dirname(os.path.abspath(self.config.persist_path)), "..", "chunk_emb_cache")
        cache_dir = os.path.abspath(cache_dir)
        cache_file = os.path.join(cache_dir, f"{key}.npy")

        if os.path.exists(cache_file):
            try:
                arr = np.load(cache_file)
                if arr.shape[0] == len(texts):
                    logger.info(f"Loaded {len(texts)} chunk embeddings from cache: {cache_file}")
                    return arr.tolist()
                logger.warning("Embedding cache size mismatch; recomputing")
            except Exception as e:
                logger.warning(f"Embedding cache read failed ({e}); recomputing")

        # Batch embedding requests to satisfy provider-side per-request limits.
        batch_size = int(getattr(self.config, "embed_batch_size", 64) or 64)
        text_embeddings = []
        for i in range(0, len(texts), batch_size):
            text_embeddings.extend(self.embedding_model._get_text_embeddings(texts[i : i + batch_size]))
        try:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_file, np.asarray(text_embeddings, dtype=np.float32))
            logger.info(f"Cached {len(texts)} chunk embeddings -> {cache_file}")
        except Exception as e:
            logger.warning(f"Embedding cache write failed ({e})")
        return text_embeddings

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
        
        # Content-addressed cache so identical chunks are not re-embedded on every
        # index rebuild (FAISS/HNSW params do not change the vectors).
        text_embeddings = self._embed_texts_cached(texts)

        vector_store = self._get_vector_store(self.config.dimensions)
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
            Settings.embed_model = self.embedding_model

            vector_store = FaissVectorStore.from_persist_dir(str(self.config.persist_path))
            faiss_index = getattr(vector_store, "_faiss_index", None) or getattr(vector_store, "faiss_index", None)
            if faiss_index is not None:
                self._apply_hnsw_runtime_params(faiss_index)
  
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

    def _update_index_from_documents(self, docs: list[Document]):
        refreshed_docs = self._index.refresh_ref_docs(docs)

        # the number of docs that are refreshed. if True in refreshed_docs, it means the doc is refreshed.
        logger.info("refresh index size is {}".format(len([True for doc in refreshed_docs if doc])))

    def _get_index(self):
        Settings.embed_model = self.embedding_model
        vector_store = self._get_vector_store(self.config.dimensions)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return  VectorStoreIndex(
            nodes = [],
            storage_context=storage_context,
            embed_model= self.embedding_model,
        )   
   

    def _similarity_score(self, object_q, object_d):
        # For llama_index based vector database, we do not need it now!
        pass

   
