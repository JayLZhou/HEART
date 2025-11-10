from pydantic import BaseModel, Field
from Config.SearchSpace import *
from Config.TopKConfig import TopK
import typing as T
from Common.Constants import DEFAULT_LLMS

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
        # ▼▼▼【修改点】在这里添加新的Reranker模型名称 ▼▼▼
        default_factory=lambda: DEFAULT_LLMS + ["ColbertRanker", "Flashrank", "Echorank"],
        description="List of LLMs or reranker models to be used for reranking.",
    )

    def defaults(self, prefix: str = "") -> T.Dict[str, T.Any]:
        return {
            f"{prefix}reranker_llm_name": self.llms[0],
            **self.top_k.defaults(prefix=f"{prefix}reranker_"),
        }

    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        return {
            f"{prefix}reranker_llm_name": CategoricalDistribution(self.llms),
            **self.top_k.build_distributions(prefix=f"{prefix}reranker_"),
        }

    def get_cardinality(self) -> int:
        return len(self.llms) * self.top_k.get_cardinality()
