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




def get_reranker(params):
    """Factory method to create a reranker instance"""
    if params["reranker_name"] not in METHOD_MAP:
        raise Exception(f"Unknown reranker method: {params['reranker_name']}.")
    return METHOD_MAP[params["reranker_name"]](method=params["reranker_name"])


