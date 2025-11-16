
import typing as T
from Rerank.Reranking import Reranker
from Rerank.Utils import MODEL_NAME_DEFAULTS



def get_reranker(params: T.Dict[str, T.Any]):
    reranker_name = params.get("reranker_name") or params.get("reranker_model", "flashrank")
    reranker_top_k = params.get("reranker_top_k", 5)
    if not params.get("reranker_model_name"):
        params["reranker_model_name"] = MODEL_NAME_DEFAULTS[reranker_name]
    return Reranker(reranker_name=reranker_name, model_name=params["reranker_model_name"], top_n=reranker_top_k)
   
