from pydantic import BaseModel, Field
from Config.TopKConfig import TopK
import typing as T
from Config.SearchSpaceMix import *
class Reranker(BaseModel, SearchSpaceMixin):
    """
    Params:
        reranker_llm_name
        reranker_top_k
    """

    top_k: TopK = Field(
        default_factory=lambda: TopK(kmax=128, log=True),
        description="Configuration for the number of items to rerank.",
    )
    llms: T.List[str] = Field(
       
        default_factory=lambda: ["upr", "flashrank", "monot5", "rankt5", "listt5", "transformer_ranker", "colbert_ranker", "twolar", "echorank", "monobert_ranker", "inranker"],
        description="List of LLMs or reranker models to be used for reranking.",
    )

    def defaults(self, prefix: str = "") -> T.Dict[str, T.Any]:
        return {
            f"{prefix}reranker_name": self.llms[0],
            **self.top_k.defaults(prefix=f"{prefix}"),
        }

    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        return {
            f"{prefix}reranker_name": CategoricalDistribution(self.llms),
            **self.top_k.build_distributions(prefix=f"{prefix}reranker_"),
        }

    def get_cardinality(self) -> int:
        return len(self.llms) * self.top_k.get_cardinality()
