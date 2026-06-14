from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
import typing as T
from Common.Constants import TEMPLATE_NAMES, DEFAULT_LLMS
from Config.FaissConfig import FaissSearchSpace
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
    faiss: FaissSearchSpace = Field(
        default_factory=FaissSearchSpace,
        description="Search-space configuration for FAISS HNSW parameters.",
    )
    synthesis_modes: T.List[str] = Field(
        default_factory=lambda: ["direct", "map_reduce", "refine"],
        description="Response synthesis strategies.",
    )
    intermediate_length_min: int = Field(default=50, description="Min tokens for intermediate answers.")
    intermediate_length_max: int = Field(default=300, description="Max tokens for intermediate answers.")
    intermediate_length_step: int = Field(default=50, description="Step for intermediate_length.")

    _custom_defaults: dict[str, T.Any] = PrivateAttr(default_factory=dict)

    def _defaults(self) -> ParamDict:
        base: ParamDict = {
            "template_name": self.template_names[0],
            "response_synthesizer_llm": self.response_synthesizer_llms[0],
            **self.rag_retriever.defaults(),
            **self.faiss.defaults(),
        }
        merged = {**base, **self._custom_defaults}
        return T.cast(ParamDict, merged)

    def update_defaults(self, defaults: ParamDict) -> None:
        self._custom_defaults.update(defaults)


    def param_names(
        self, params: T.Dict[str, T.Any] | T.List[str] | None = None
    ) -> T.List[str]:
        return list(self.build_distributions(params=params).keys())

    def build_distributions(
        self, params: T.Dict[str, T.Any] | T.List[str] | None = None
    ) -> T.Dict[str, BaseDistribution]:
        # None => full introspection (e.g. param_names()): include optional FAISS + reranker blocks.
        if params is None:
            param_names = {"faiss", "reranker"}
        else:
            param_names = set(params)
        distributions: dict[str, BaseDistribution] = {
            "template_name": CategoricalDistribution(self.template_names),
            "response_synthesizer_llm": CategoricalDistribution(
                self.response_synthesizer_llms
            ),
            "synthesis_mode": CategoricalDistribution(self.synthesis_modes),
            "intermediate_length": IntDistribution(
                self.intermediate_length_min,
                self.intermediate_length_max,
                step=self.intermediate_length_step,
            ),
        }

        distributions.update(self.rag_retriever.build_distributions())
        if "faiss" in param_names:
            distributions.update(self.faiss.build_distributions())
        if "reranker" in param_names:
            distributions['reranker'] = self.reranker.build_distributions()
      


        # if params is not None:
        #     reduced_distributions = {
        #         key: val for key, val in distributions.items() if key in params
        #     }
        #     return reduced_distributions
   
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

        params['rag_retriever'] = self.rag_retriever.sample(trial)
        if "faiss" in parameters:
            params.update(self.faiss.sample(trial))
        else:
            params.update(self.faiss.defaults())
        
        if "reranker" in parameters:
                params['reranker'] = self.reranker.sample(trial)
                params['reranker']["reranker_enabled"] = True
        else:
            params['reranker'] = self.reranker.defaults()
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
        )
    
        sub_card *= self.rag_retriever.get_cardinality()
        sub_card *= self.faiss.get_cardinality()

        card += sub_card

        return card
