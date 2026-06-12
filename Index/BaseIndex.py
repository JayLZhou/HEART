import os
import threading
from abc import ABC, abstractmethod
from Common.Utils import clean_storage
from Common.Logger import logger

_index_build_lock = threading.Lock()

class BaseIndex(ABC):
    def __init__(self, config):
        self.config = config
        self._index = None

    def build_index(self, elements, meta_data, force=False):
        with _index_build_lock:
            logger.info("Starting insert elements of the given graph into vector database")
     
            from_load = False
            if self.exist_index() and not force:
                logger.info("Loading index from the file {}".format(self.config.persist_path))
                from_load = self._load_index()
            else:
            
                self._index = self._get_index()
            if not from_load:
                # Note: When you successfully load the index from a file, you don't need to rebuild it.
                self.clean_index()
                logger.info("Building index for input elements")
                self._update_index(elements, meta_data)
                self._storage_index()
                logger.info("Index successfully built and stored.")
            logger.info("✅ Finished starting insert entities of the given graph into vector database")

    def exist_index(self):
        return os.path.exists(self.config.persist_path)

    @abstractmethod
    def retrieval(self, query, top_k):
        pass

    @abstractmethod
    def _get_index(self):
        pass

    @abstractmethod
    def retrieval_batch(self, queries, top_k):
        pass

    @abstractmethod
    def _update_index(self, elements, meta_data):
        pass

    @abstractmethod
    def _get_retrieve_top_k(self):
        return 10

    @abstractmethod
    def _storage_index(self):
        pass

    @abstractmethod
    def _load_index(self) -> bool:
        pass

    def similarity_score(self, object_q, object_d):
        return self._similarity_score(object_q, object_d)

    def _similarity_score(self, object_q, object_d):
        pass

    def get_max_score(self, query):
        pass

    def clean_index(self):
       clean_storage(self.config.persist_path)
