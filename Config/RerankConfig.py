from pydantic import BaseModel, Field
from Config.TopKConfig import TopK
import typing as T
from Config.SearchSpaceMix import *
class Reranker(BaseModel, SearchSpaceMixin):
    top_k: TopK = Field(
        default_factory=lambda: TopK(kmax=128, log=True),
    )
    choices: T.List[str] = Field(
        default_factory=lambda: [
            "flashrank::ms-marco-TinyBERT-L-2-v2",
            "flashrank::ms-marco-MiniLM-L-12-v2",
            "qwen_reranker::qwen3-reranker-0.6b",
            "transformer_ranker::mxbai-rerank-base",
            "transformer_ranker::bge-reranker-v2-m3",
            "transformer_ranker::jina-reranker-base-multilingual",
            "transformer_ranker::gte-multilingual-reranker-base",
            "upr::t5-base",
        ],
    )

    def defaults(self, prefix: str = "") -> T.Dict[str, T.Any]:
        return {
            f"{prefix}reranker_choice": self.choices[0],
            **self.top_k.defaults(prefix=f"{prefix}"),
        }

    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        return {
            f"{prefix}reranker_choice": CategoricalDistribution(self.choices),
            **self.top_k.build_distributions(prefix=f"{prefix}reranker_"),
        }

    def get_cardinality(self) -> int:
        return len(self.choices) * self.top_k.get_cardinality()
