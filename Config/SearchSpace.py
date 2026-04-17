from pydantic import BaseModel, ConfigDict, Field
import typing as T
from Common.Constants import TEMPLATE_NAMES, DEFAULT_LLMS
from Config.RetrieverConfig import Retriever
from Config.TopKConfig import TopK
from Config.RerankConfig import Reranker
from Config.SearchSpaceMix import *
import numpy as np
import random
from optuna.distributions import (
    FloatDistribution,
    IntDistribution,
    CategoricalDistribution,
    BaseDistribution,
)


class SearchSpace(BaseModel):
    model_config = ConfigDict(extra="forbid")  # Forbids unknown fields
    non_search_space_params: T.List[str] = Field(
        default_factory=lambda: [
            "enforce_full_evaluation",
            "retrievers",
        ],
        description="Parameters not part of the hyperparameter search space.",
    )

    template_names: T.List[str] = Field(
        default_factory=lambda: TEMPLATE_NAMES,
        description="List of available prompt template names.",
    )
    response_synthesizer_llms: T.List[str] = Field(
        default_factory=lambda: DEFAULT_LLMS,
        description="List of LLMs for response synthesis.",
    )

    rag_retriever: Retriever = Field(
        default_factory=lambda: Retriever(top_k=TopK(kmax=128, log=True)),
        description="Configuration for the RAG retriever.",
    )


    reranker: Reranker = Field(
        default_factory=Reranker, description="Configuration for the reranker."
    )
    faiss_hnsw_m_values: T.List[int] = Field(
        default_factory=lambda: [16, 24, 32, 40, 48, 56, 64],
        description="Candidate HNSW M values for FAISS HNSW index.",
    )
    faiss_hnsw_ef_search_values: T.List[int] = Field(
        default_factory=lambda: [32, 64, 96, 128, 160, 192, 224, 256],
        description="Candidate HNSW efSearch values for FAISS HNSW index.",
    )
    faiss_hnsw_ef_construction_values: T.List[int] = Field(
        default_factory=lambda: [40, 80, 120, 160, 200, 240, 280, 320],
        description="Candidate HNSW efConstruction values for FAISS HNSW index.",
    )
    faiss_metric_values: T.List[str] = Field(
        default_factory=lambda: ["l2", "inner_product"],
        description="Candidate FAISS metric values.",
    )




    def _defaults(self) -> ParamDict:
        return {
            "template_name": self.template_names[0],
            "response_synthesizer_llm": self.response_synthesizer_llms[0],
            "faiss_hnsw_m": self.faiss_hnsw_m_values[0],
            "faiss_hnsw_ef_search": self.faiss_hnsw_ef_search_values[0],
            "faiss_hnsw_ef_construction": self.faiss_hnsw_ef_construction_values[0],
            "faiss_metric": self.faiss_metric_values[0],
            **self.rag_retriever.defaults(),
        }

    def update_defaults(self, defaults: ParamDict) -> None:
        self._custom_defaults.update(defaults)


    def param_names(
        self, params: T.Dict[str, T.Any] | T.List[str] | None = None
    ) -> T.List[str]:
        return list(self.build_distributions(params=params).keys())

    def build_distributions(
        self, params: T.Dict[str, T.Any] | T.List[str] | None = None
    ) -> T.Dict[str, BaseDistribution]:
        distributions: dict[str, BaseDistribution] = {
            "template_name": CategoricalDistribution(self.template_names),
            "response_synthesizer_llm": CategoricalDistribution(
                self.response_synthesizer_llms
            ),
            "faiss_hnsw_m": CategoricalDistribution(self.faiss_hnsw_m_values),
            "faiss_hnsw_ef_search": CategoricalDistribution(self.faiss_hnsw_ef_search_values),
            "faiss_hnsw_ef_construction": CategoricalDistribution(self.faiss_hnsw_ef_construction_values),
            "faiss_metric": CategoricalDistribution(self.faiss_metric_values),
          
        }

        distributions.update(self.rag_retriever.build_distributions())
        if params is not None and "reranker" in params:
            distributions['reranker'] = self.reranker.build_distributions()

        return distributions

    def sample(self, trial: Trial, parameters: T.List[str]) -> ParamDict:
        if not hasattr(self, "_default_params"):
            self._default_params = self._defaults()
        params: ParamDict = {}
        if "template_name" in parameters:
            params["template_name"] = trial.suggest_categorical(
                "template_name", self.template_names
            )
        else:
            params["template_name"] = self._default_params["template_name"]

        if "response_synthesizer_llm" in parameters:

            params["response_synthesizer_llm"] = trial.suggest_categorical(
                "response_synthesizer_llm", self.response_synthesizer_llms
            )
        else:
            params["response_synthesizer_llm"] = self._default_params["response_synthesizer_llm"]

        for name, values in (
            ("faiss_hnsw_m", self.faiss_hnsw_m_values),
            ("faiss_hnsw_ef_search", self.faiss_hnsw_ef_search_values),
            ("faiss_hnsw_ef_construction", self.faiss_hnsw_ef_construction_values),
            ("faiss_metric", self.faiss_metric_values),
        ):
            if name in parameters:
                params[name] = trial.suggest_categorical(name, values)
            else:
                params[name] = self._default_params[name]

        params['rag_retriever'] = self.rag_retriever.sample(trial)
       
        if "reranker" in parameters:
                params['reranker'] = self.reranker.sample(trial)
                params['reranker']["reranker_enabled"] = True
        else:
            params['reranker']["reranker_enabled"] = False


        return params
    

    def sample_from_distributions(
        self, dists: T.Dict[str, BaseDistribution]
    ) -> T.Dict[str, T.Any]:
        """
        Randomly sample values given a dict: {param_name: BaseDistribution}.
        Supports Float, Int, Categorical.
        """

        sample = {}

        for name, dist in dists.items():
            if isinstance(dist, FloatDistribution):
                if dist.log:
                    # sample log-uniform
                    v = np.exp(
                        random.uniform(np.log(dist.low), np.log(dist.high))
                    )
                else:
                    v = random.uniform(dist.low, dist.high)
                sample[name] = v

            elif isinstance(dist, IntDistribution):
                if dist.log:
                    # log-uniform integer
                    v = int(
                        np.exp(
                            random.uniform(np.log(dist.low), np.log(dist.high))
                        )
                    )
                else:
                    # integers are inclusive
                    v = random.randint(dist.low, dist.high)
                sample[name] = v

            elif isinstance(dist, CategoricalDistribution):
                sample[name] = random.choice(dist.choices)

            else:
                raise NotImplementedError(f"Unsupported distribution type: {dist}")

        return sample

    def get_cardinality(self) -> int:
        card = 0

        sub_card = (
            len(self.template_names)
            * len(self.response_synthesizer_llms)
            * len(self.faiss_hnsw_m_values)
            * len(self.faiss_hnsw_ef_search_values)
            * len(self.faiss_hnsw_ef_construction_values)
            * len(self.faiss_metric_values)
        )
    
        sub_card *= self.rag_retriever.get_cardinality()

        card += sub_card

        return card
