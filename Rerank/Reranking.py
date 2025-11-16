from Rerank.Upr import UPR
from Rerank.ColbertRanker import ColBERTReranker
from Rerank.Flashrank import FlashRanker
from Rerank.Monot5 import MonoT5
from Rerank.Rankt5 import RankT5
from Rerank.Echorank import EchoRankReranker
from Rerank.Listt5 import ListT5
from Rerank.Twolar import TWOLAR
from Rerank.TransformerRanker import TransformerRanker
from Rerank.Monobert import MonoBERT
from Rerank.InRanker import InRanker
import typing as T
from Rerank.Utils import HF_PRE_DEFIND_MODELS
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from Common.Logger import logger
METHOD_MAP = {
    # Existing reranking methods
    'upr': UPR,
    'flashrank': FlashRanker,
    'monot5': MonoT5,
    'rankt5': RankT5,
    'listt5': ListT5,
    'transformer_ranker': TransformerRanker,
    'colbert_ranker': ColBERTReranker,
    'twolar': TWOLAR,
    'echorank': EchoRankReranker,
    'monobert_ranker': MonoBERT,
    "inranker": InRanker
}

# Create dummy classes for type hints when user's framework is not available
class Document:
    def __init__(self, question=None, contexts=None):
        self.question = question
        self.contexts = contexts or []
        self.reorder_contexts = []

class Question:
    def __init__(self, question):
        self.question = question
        
class Context:
    def __init__(self, text, id, score=0.0):
        self.text = text
        self.id = id
        self.score = score

class Reranker(BaseNodePostprocessor):
    """
    Adapter to integrate user's reranker models with LlamaIndex's postprocessor interface.
    This bridges the gap between user's Document/Context schema and LlamaIndex's NodeWithScore.
    """
    
    def __init__(self, reranker_name: str, model_name: str = None, top_n: int = 5, **kwargs):
        super().__init__()
        self._reranker_name = reranker_name.lower()
        self._model_name = model_name
        self._top_n = top_n
        self._kwargs = kwargs
        self._initialize_reranker()
        
    def _initialize_reranker(self):
        try:
            if self._reranker_name not in METHOD_MAP:
                raise ValueError(f"Unknown reranker: {self._reranker_name}. Available: {list(METHOD_MAP.keys())}")
            
            self._reranker = METHOD_MAP[self._reranker_name](method=self._reranker_name, model_name=HF_PRE_DEFIND_MODELS[self._reranker_name][self._model_name], **self._kwargs)
            logger.info(f"✅ Successfully initialized {self._reranker_name} reranker with model {self._model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize {self._reranker_name} reranker: {e}")
            self._reranker = None
            raise



    def _postprocess_nodes(
        self,
        nodes: T.List[NodeWithScore],
        query_bundle: T.Optional[QueryBundle] = None,
    ) -> T.List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Query bundle is required for reranking.")
        if not nodes:
            return []
        
        if self._reranker is None:
            logger.error(f"❌ Reranker not initialized for {self._reranker_name}")
            return nodes[:self._top_n]  # Fallback to original order

        try:
            # Convert to user's format
            document = self._convert_to_user_format(nodes, query_bundle.query_str)
            
            # Apply user's reranker
            reranked_documents = self._reranker.rank([document])
            
            # Convert back to LlamaIndex format
            return self._convert_from_user_format(reranked_documents[0], nodes)
            
        except Exception as e:
            logger.error(f"❌ Reranking failed with {self._reranker_name}: {e}")
            return nodes[:self._top_n]  # Fallback to original order



