"""
Index Config Factory.
"""
from Index import get_rag_embedding
from Index.Schema import (
    VectorIndexConfig,
    ColBertIndexConfig,
    FAISSIndexConfig,
    BMIndexConfig
)


class IndexConfigFactory:
    def __init__(self):
        self.dense_creators = {
            "vector": self._create_vector_config,
            "colbert": self._create_colbert_config,
            "faiss": self._create_faiss_config,
        }
        self.sparse_creators = {
            "bm25": self._create_bm25_config,
        }
    def get_config(self, config, persist_path, type="dense"):
        """Key is PersistType."""
        if type == "dense":
            return self.dense_creators[config.vdb_type](config, persist_path)
        elif type == "sparse":
            return self.sparse_creators[config.sparse_index_type](config, persist_path)
    @staticmethod
    def _create_vector_config(config, persist_path):
        return VectorIndexConfig(
            persist_path=persist_path,
            embed_model=get_rag_embedding(config.embedding.api_type, config)
        )

    @staticmethod
    def _create_faiss_config(config, persist_path):
        return FAISSIndexConfig(
            persist_path=persist_path,
            embed_model=get_rag_embedding(config.embedding.api_type, config),
            dimensions=config.embedding.dimensions,
            hnsw_m=config.faiss_hnsw_m,
            hnsw_ef_search=config.faiss_hnsw_ef_search,
            hnsw_ef_construction=config.faiss_hnsw_ef_construction,
            metric=config.faiss_metric,
        )

    @staticmethod
    def _create_colbert_config(config, persist_path):
        return ColBertIndexConfig(persist_path=persist_path, index_name="nbits_2",
                                  model_name=config.colbert_checkpoint_path, nbits=2)

    @staticmethod

    def _create_bm25_config(config, persist_path):

        return  BMIndexConfig(
                persist_path=persist_path,
                **config.dict()
            )
        
get_index_config = IndexConfigFactory().get_config
