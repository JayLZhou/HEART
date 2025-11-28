#!/usr/bin/env python
# -*- coding: utf-8 -*-
import typing as T

from Common.Constants import DEFAULT_LLMS
from pydantic import BaseModel, Field
from optuna import Trial
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    IntDistribution,
    FloatDistribution,
)
from Config.SearchSpaceMix import SearchSpaceMixin, ParamDict, get_dist_cardinality
from Config.TopKConfig import TopK



class Hybrid(BaseModel, SearchSpaceMixin):
    bm25_weight_min: float = Field(
        default=0.1, description="Minimum weight for BM25 in hybrid retrieval."
    )
    bm25_weight_max: float = Field(
        default=0.9, description="Maximum weight for BM25 in hybrid retrieval."
    )
    bm25_weight_step: float = Field(
        default=0.1, description="Step size for BM25 weight."
    )

    def defaults(self, prefix: str = "") -> T.Dict[str, T.Any]:
        return {
            f"{prefix}hybrid_bm25_weight": 0.5,
        }

    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        bm25_weight = f"{prefix}hybrid_bm25_weight"
        return {
            bm25_weight: FloatDistribution(
                self.bm25_weight_min, self.bm25_weight_max, step=self.bm25_weight_step
            ),
        }

    def get_cardinality(self) -> int:
        return get_dist_cardinality(
            self.bm25_weight_min, self.bm25_weight_max, self.bm25_weight_step
        )

class QueryDecomposition(BaseModel, SearchSpaceMixin):
    llm_names: T.List[str] = Field(
        default_factory=lambda: DEFAULT_LLMS,
        description="List of LLM names to be used for query decomposition.",
    )
    num_queries_min: int = Field(
        default=2, description="Minimum number of sub-queries to generate."
    )
    num_queries_max: int = Field(
        default=20, description="Maximum number of sub-queries to generate."
    )
    num_queries_step: int = Field(
        default=2, description="Step size for the number of sub-queries."
    )

    def defaults(self, prefix: str = "") -> T.Dict[str, T.Any]:
        return {
            f"{prefix}query_decomposition_enabled": False,
        }

    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        return {
            f"{prefix}query_decomposition_llm_name": CategoricalDistribution(
                self.llm_names
            ),
            f"{prefix}query_decomposition_num_queries": IntDistribution(
                self.num_queries_min,
                self.num_queries_max,
                step=self.num_queries_step,
            ),
        }

    def get_cardinality(self) -> int:
        return len(self.llm_names) * get_dist_cardinality(
            self.num_queries_min, self.num_queries_max, self.num_queries_step
        )

class FusionMode(BaseModel, SearchSpaceMixin):
    fusion_modes: T.List[str] = Field(
        default_factory=lambda: [
            "simple",
            "reciprocal_rerank",
            "relative_score",
            "dist_based_score",
        ],
        description="List of available fusion modes for combining results from multiple retrievers or queries.",
    )

    def defaults(self, prefix: str = "") -> T.Dict[str, T.Any]:
        return {f"{prefix}fusion_mode": "simple"}

    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        return {
            f"{prefix}fusion_mode": CategoricalDistribution(self.fusion_modes),
        }

    def get_cardinality(self) -> int:
        return len(self.fusion_modes)


class Retriever(BaseModel, SearchSpaceMixin):
    top_k: TopK = Field(
        default_factory=TopK,
        description="Configuration for the number of items to retrieve.",
    )
    methods: T.List[str] = Field(
        default_factory=lambda: [
            "dense",
            "sparse",
            "hybrid",
        ],
        description="List of supported retrieval methods: dense (based on embedding models), sparse (BM25) or hybrid.",
    )
    hybrid: Hybrid = Field(
        default_factory=Hybrid, description="Configuration for hybrid retrieval."
    )
    query_decomposition_enabled: list[bool] = Field(
        default_factory=lambda: [True, False],
        description="Whether query decomposition is enabled.",
    )
    query_decomposition: QueryDecomposition = Field(
        default_factory=QueryDecomposition,
        description="Configuration for query decomposition.",
    )
    fusion: FusionMode = Field(
        default_factory=FusionMode, description="Configuration for fusing results."
    )

    def defaults(self, prefix: str = "rag_") -> ParamDict:
        params = {
            f"{prefix}method": "dense",
            **self.top_k.defaults(prefix=prefix),
            **self.query_decomposition.defaults(prefix=prefix),
            **self.hybrid.defaults(prefix=prefix),
            **self.fusion.defaults(prefix=prefix),
        }
        return T.cast(ParamDict, params)

    def build_distributions(
        self, prefix: str = "rag_"
    ) -> T.Dict[str, BaseDistribution]:
        distributions = {
            f"{prefix}method": CategoricalDistribution(self.methods),
            f"{prefix}query_decomposition_enabled": CategoricalDistribution(
                self.query_decomposition_enabled
            ),
            **self.top_k.build_distributions(prefix=prefix),
        }

        if "hybrid" in self.methods:
            distributions.update(**self.hybrid.build_distributions(prefix=prefix))

        if True in self.query_decomposition_enabled:
            distributions.update(
                **self.query_decomposition.build_distributions(prefix=prefix)
            )

        if "hybrid" in self.methods or True in self.query_decomposition_enabled:
            distributions.update(**self.fusion.build_distributions(prefix=prefix))

        return distributions

    def sample(self, trial: Trial, prefix: str = "") -> ParamDict:
        method = f"{prefix}method"
        use_query_decomp = f"{prefix}query_decomposition_enabled"
   
        params = {
            method: trial.suggest_categorical(method, self.methods),
            use_query_decomp: trial.suggest_categorical(
                use_query_decomp, self.query_decomposition_enabled
            ),
            **self.top_k.sample(trial, prefix=prefix),
        }

        params[method] = "hybrid" # for debugging
        if params[method] == "hybrid":
            params.update(**self.hybrid.sample(trial, prefix=prefix))

        if params[use_query_decomp]:
            params.update(**self.query_decomposition.sample(trial, prefix=prefix))

        if params[method] == "hybrid" or params[use_query_decomp]:
            params.update(**self.fusion.sample(trial, prefix=prefix))

        return T.cast(ParamDict, params)

    def get_cardinality(self) -> int:
        card = (
            self.top_k.get_cardinality()
            * len(self.methods)
            * len(self.query_decomposition_enabled)
        )
        if "hybrid" in self.methods:
            card *= self.hybrid.get_cardinality()
        if True in self.query_decomposition_enabled:
            card *= self.query_decomposition.get_cardinality()
        if "hybrid" in self.methods or True in self.query_decomposition_enabled:
            card *= self.fusion.get_cardinality()
        return card






