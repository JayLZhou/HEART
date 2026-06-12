import typing as T

from Rerank.Reranking import Reranker
from Rerank.Utils import MODEL_NAME_DEFAULTS


def _parse_reranker_choice(choice: str) -> tuple[str, str]:
    if "::" not in choice:
        raise ValueError(
            f"Invalid reranker_choice '{choice}'. Expected format 'reranker_name::reranker_model_name'."
        )
    return choice.split("::", 1)


def get_reranker(params: T.Dict[str, T.Any]):
    reranker_choice = params.get("reranker_choice")
    if reranker_choice:
        reranker_name, reranker_model_name = _parse_reranker_choice(reranker_choice)
        params["reranker_name"] = reranker_name
        params["reranker_model_name"] = reranker_model_name
    else:
        reranker_name = params.get("reranker_name") or params.get("reranker_model", "flashrank")
        reranker_model_name = params.get("reranker_model_name")
        if reranker_model_name:
            params["reranker_model_name"] = reranker_model_name

    reranker_top_k = params.get("reranker_top_k", 5)
    if not params.get("reranker_model_name"):
        params["reranker_model_name"] = MODEL_NAME_DEFAULTS[reranker_name]

    return Reranker(
        reranker_name=reranker_name,
        model_name=params["reranker_model_name"],
        top_n=reranker_top_k,
    )
