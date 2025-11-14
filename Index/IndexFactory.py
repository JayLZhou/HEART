
from Common.BaseFactory import ConfigBasedFactory
from Index.Schema import (
    BaseIndexConfig,
    VectorIndexConfig,
    ColBertIndexConfig,
    FAISSIndexConfig,
    BMIndexConfig
)
from Index.VectorIndex import VectorIndex
from Index.FaissIndex import FaissIndex
from Index.BMIndex import BMIndex



class RAGIndexFactory(ConfigBasedFactory):
    def __init__(self):
        creators = {
            VectorIndexConfig: self._create_vector_index,
            ColBertIndexConfig: self._create_colbert,
            FAISSIndexConfig: self._create_faiss,
            BMIndexConfig: self._create_bm_index,
        }
        super().__init__(creators)

    def get_index(self, config: BaseIndexConfig):
        """Key is IndexType."""
        return super().get_instance(config)

    @classmethod
    def _create_vector_index(cls, config):
        return VectorIndex(config)

    @classmethod
    def _create_colbert(cls, config: ColBertIndexConfig):
        return ColBertIndex(config)

    
    def _create_faiss(self, config):
       return FaissIndex(config)

    def _create_bm_index(self, config):
        return BMIndex(config)


get_index = RAGIndexFactory().get_index
