from dataclasses import dataclass, field
from typing import Dict, Optional, List
from Common.Logger import logger
from Index.FaissIndex import FaissIndex
from Index import get_index, get_index_config
from Option.Config2 import Config
from Storage.BaseStorage import BaseStorage
from Storage.NameSpace import Namespace
import numpy as np

class QueryContent:
    content: str
    def __init__(self, text):
        self.content = text

class QueryStorage(BaseStorage):
    """Simple text → embedding → FAISS storage with id support."""

    index: FaissIndex = None
    _key_to_text: Dict[str, str] = {}     # key → text
    _key_to_metadata: Dict[str, dict] = {}  # key → metadata

    def __init__(self, config: Config, namespace: Namespace):
        self.config = config
        self.namespace = namespace
        self.index = get_index(get_index_config(config, self.namespace.workspace.make_for("query_vdb").get_save_path()))

    def size(self):
        return len(self._key_to_text)

    def _get_query_key(self, query: dict):
        key = query.get("_id", query.get("id"))
        if key is None:
            return None
        return str(key)

    # --------------------------
    # INSERT
    # --------------------------
    def upsert(self, query: dict):
        """Insert a text document. Automatically embeds and store in FAISS."""

        # 记录文本与 metadata
        key, text = self._get_query_key(query), query["question"]
        if key is None:
            raise KeyError("Query must contain either '_id' or 'id'")
        self._key_to_text[key] = text
        self._key_to_metadata[key] = query

        # 构造 FaissIndex 需要的 datas 格式
        # key 为唯一 id
        data_item = {
            0: key,                  # id
            1: QueryContent(text),     # 伪装成你的 TextChunk/document
            **query
        }

        self.index._update_index([data_item], meta_data=list(query.keys()))
        logger.info(f"Inserted text key={key}")

    # --------------------------
    # BATCH INSERT
    # --------------------------
    def upsert_batch(self, queries: List[dict]):
        """
        items: list of (key, text, metadata)
        """
        datas = []
        meta_keys = set()

        for query in queries:

            key, text = self._get_query_key(query), query["question"]
            if key is None:
                raise KeyError("Query must contain either '_id' or 'id'")
            self._key_to_text[key] = text
            self._key_to_metadata[key] = query

            # 构造 FaissIndex 需要的 datas 格式
            # key 为唯一 id
            datas.append({
                0: key,                  # id
                1: QueryContent(text),     # 伪装成你的 TextChunk/document
                **query
            })
            meta_keys.update(query.keys())

        self.index._update_index(datas, meta_data=list(meta_keys))
        logger.info(f"Batch inserted {len(queries)} items")

    # --------------------------
    # RETRIEVAL
    # --------------------------
    def query(self, query_text: str, top_k: int = 5):
        """Return top-k similar texts and their metadata."""
        results = self.index.retrieval(query_text, top_k=top_k)

        final = []
        for r in results:
            key = r.node.metadata.get("_id", r.node.metadata.get("id"))
            score = r.score
            text = self._key_to_text.get(key, "")

            final.append({
                "key": key,
                "text": text,
                "metadata": self._key_to_metadata.get(key, {}),
                "score": score,
            })

        return final

    # --------------------------
    # DELETE
    # --------------------------
    def delete(self, key: str):
        """Delete a text item (does not remove from FAISS; you can rebuild if needed)."""
        if key not in self._key_to_text:
            logger.warning(f"Key {key} not found")
            return

        del self._key_to_text[key]
        del self._key_to_metadata[key]

        logger.warning("Note: FAISS does not support deletion. Need rebuild to remove effect.")

    # --------------------------
    # HELPERS
    # --------------------------
    def get_text(self, key: str) -> Optional[str]:
        return self._key_to_text.get(key)

    def get_metadata(self, key: str) -> Optional[dict]:
        return self._key_to_metadata.get(key)
    


if __name__ == "__main__":
    from Option.Config2 import Config
    import argparse
    import os
    import random
    import numpy as np
    import torch
    from pathlib import Path
    from shutil import copyfile
    from Data.DataLoader import RAGDataset
    from Common.Utils import welcome_message
    from tqdm import tqdm
    from Common.Logger import logger
    from Pipeline.FlowBuild import FlowBuilder
    from Tuner.TunerFactory import get_tuner
    from Utils.Evaluation import Evaluator
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
    parser.add_argument("-dataset_name", type=str, help="Name of the dataset.")
    args = parser.parse_args()

    opt = Config.parse(Path(args.opt), dataset_name=args.dataset_name)
    qs = QueryStorage(opt)

    # 插入 3 条数据
    qs.upsert({"_id": "q1", "question": "What is apple?", "tag": "fruit"})
    qs.upsert({"_id": "q2", "question": "What is a banana?", "tag": "fruit"})
    qs.upsert({"_id": "q3", "question": "How fast is a car?", "tag": "vehicle"})

    # 查询
    print("\n=== Query: 'Tell me about fruits' ===")
    res = qs.query("Tell me about fruits", top_k=2)

    for r in res:
        print(r)