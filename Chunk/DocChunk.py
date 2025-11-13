from concurrent.futures import ProcessPoolExecutor, as_completed
from Chunk.ChunkFactory import create_chunk_method
from Common.Utils import mdhash_id
from Common.Logger import logger
from Schema.ChunkSchema import TextChunk
from Storage.ChunkKVStorage import ChunkKVStorage
from typing import List, Union
import tiktoken
import os

class DocChunk:
    def __init__(self, config, token_model, namesapce):
        self.config = config
        self.chunk_method = create_chunk_method(self.config.chunk_method)
        self._chunk = ChunkKVStorage(namespace=namesapce)
        self.token_model = tiktoken.encoding_for_model(token_model)

    @property
    def namespace(self):
        return None

    # TODO: Try to rewrite here, not now
    @namespace.setter
    def namespace(self, namespace):
        self.namespace = namespace

    def build_chunks(self, docs: Union[str, List[str]], force=True):
        logger.info("Starting chunk the given documents")
  
        is_exist = self._load_chunk(force)
        if not is_exist or force:

            # TODO: Now we only support the str, list[str], Maybe for more types.
            if isinstance(docs, str):
                docs = [docs]

            if isinstance(docs, list):
                if all(isinstance(doc, dict) for doc in docs):
                    docs = {
                        mdhash_id(doc["content"].strip(), prefix="doc-"): {
                            "content": doc["content"].strip(),
                            "title": doc.get("title", ""),
                        }
                        for doc in docs
                    }
                else:
                    docs = {
                        mdhash_id(doc.strip(), prefix="doc-"): {
                            "content": doc.strip(),
                            "title": "",
                        }
                        for doc in docs
                    }

            flatten_list = list(docs.items())
            docs = [doc[1]["content"] for doc in flatten_list]
            doc_keys = [doc[0] for doc in flatten_list]
            title_list = [doc[1]["title"] for doc in flatten_list]
            tokens = self.token_model.encode_batch(docs, num_threads=16)

            chunks = self.chunk_method(
                tokens,
                doc_keys=doc_keys,
                tiktoken_model=self.token_model,
                title_list=title_list,
                overlap_token_size=self.config.chunk_overlap_token_size,
                max_token_size=self.config.chunk_token_size,
            )

            for chunk in chunks:
                chunk["chunk_id"] = mdhash_id(chunk["content"], prefix="chunk-")
                self._chunk.upsert(chunk["chunk_id"], TextChunk(**chunk))

            self._chunk.persist()
        logger.info("✅ Finished the chunking stage")

    def _load_chunk(self, force=False):
        if force:
            return False
        return self._chunk.load_chunk()

    def get_chunks(self):
        return self._chunk.get_chunks()

    def get_index_by_merge_key(self, chunk_id):
        return  self._chunk.get_index_by_merge_key(chunk_id)

    @property
    def size(self):
        return  self._chunk.size()

    def get_index_by_key(self, key):
        return  self._chunk.get_index_by_key(key)

    def get_data_by_key(self, chunk_id):

        chunk =  self._chunk.get_by_key(chunk_id)
        return chunk.content

    def get_data_by_index(self, index):
        chunk =  self._chunk.get_data_by_index(index)
        return chunk.content

    def get_key_by_index(self, index):
        return  self._chunk.get_key_by_index(index)

    def get_data_by_indices(self, indices):
        """Get data by multiple indices using multiprocessing"""
        if not indices:
            return []
        
        # Extract data dictionary for worker processes
        data_dict = self._chunk._data
        
        # Use ProcessPoolExecutor for parallel processing
        max_workers = min(len(indices), os.cpu_count() or 4)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks - pass data_dict and index
            future_to_index = {
                executor.submit(_get_data_by_index_worker, data_dict, index): index 
                for index in indices
            }
            
            # Collect results in order
            results = [None] * len(indices)
            index_to_position = {idx: pos for pos, idx in enumerate(indices)}
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index_to_position[index]] = result
                except Exception as e:
                    logger.error(f"Error getting data for index {index}: {e}")
                    results[index_to_position[index]] = None
        
        return results


def _get_data_by_index_worker(data_dict, index):
    """Worker function for multiprocessing - must be at module level for pickling"""
    chunk = data_dict.get(index, None)
    if chunk is None:
        return None
    return chunk.content
