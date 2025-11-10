import typing as T
from abc import ABC, abstractmethod
from optuna import Trial
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    DiscreteUniformDistribution,
    FloatDistribution,
    IntDistribution,
    LogUniformDistribution,
    UniformDistribution,
)
from pydantic import ConfigDict
from Common.Constants import NDIGITS
ParamDict = T.Dict[str, str | int | float | bool]
from Common.Utils import get_dist_cardinality

class SearchSpaceMixin(ABC):
    """Common interface for all search space classes."""

    model_config = ConfigDict(extra="forbid")  # Forbids unknown fields

    @abstractmethod
    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        """Subclasses must return the distributions defining their parameter search space."""
        pass

    @abstractmethod
    def get_cardinality(self) -> int:
        """Subclasses must define a method to compute the cardinality of their space."""
        pass

    def sample(self, trial: Trial, prefix: str = "") -> ParamDict:
        """Sample concrete parameters from the search space distributions."""
        return {
            name: self._suggest_from_distribution(trial, name, dist)
            for name, dist in self.build_distributions(prefix).items()
        }

    def _suggest_from_distribution(
        self, trial: Trial, name: str, dist: BaseDistribution
    ) -> T.Any:
        if isinstance(dist, CategoricalDistribution):
            return trial.suggest_categorical(name, dist.choices)
        elif isinstance(dist, IntDistribution):
            return trial.suggest_int(
                name, low=dist.low, high=dist.high, step=dist.step, log=dist.log
            )
        elif isinstance(dist, FloatDistribution):
            value = trial.suggest_float(
                name, low=dist.low, high=dist.high, step=dist.step, log=dist.log
            )
            return round(value, ndigits=NDIGITS)
        elif isinstance(dist, DiscreteUniformDistribution):
            value = trial.suggest_discrete_uniform(
                name, low=dist.low, high=dist.high, q=dist.q
            )
            return round(value, ndigits=NDIGITS)
        elif isinstance(dist, LogUniformDistribution):
            value = trial.suggest_loguniform(name, low=dist.low, high=dist.high)
            return round(value, ndigits=NDIGITS)
        elif isinstance(dist, UniformDistribution):
            value = trial.suggest_uniform(name, low=dist.low, high=dist.high)
            return round(value, ndigits=NDIGITS)
        else:
            raise NotImplementedError(f"Unsupported distribution type: {type(dist)}")


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

    reranker_enabled: T.List[bool] = Field(
        default_factory=lambda: [True, False],
        description="Whether reranking is enabled.",
    )
    reranker: Reranker = Field(
        default_factory=Reranker, description="Configuration for the reranker."
    )
    hyde_enabled: T.List[bool] = Field(
        default_factory=lambda: [True, False], description="Whether HyDE is enabled."
    )

  
    _custom_defaults: ParamDict = {}

    def _defaults(self) -> ParamDict:
        return {
            "template_name": self.template_names[0],
            "response_synthesizer_llm": self.response_synthesizer_llms[0],
            "reranker_enabled": False,
            **self.rag_retriever.defaults(),
            **self.splitter.defaults()
        }

    def update_defaults(self, defaults: ParamDict) -> None:
        self._custom_defaults.update(defaults)

    def defaults(self) -> ParamDict:
        return {
            **self._defaults(),
            **self._custom_defaults,
        }

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
            "reranker_enabled": CategoricalDistribution(self.reranker_enabled),
        }

        distributions.update(self.rag_retriever.build_distributions())
        distributions.update(self.splitter.build_distributions())
        if True in self.reranker_enabled:
            distributions.update(self.reranker.build_distributions())


        if params is not None:
            reduced_distributions = {
                key: val for key, val in distributions.items() if key in params
            }
            return reduced_distributions

        return distributions

    def sample(self, trial: Trial, parameters: T.List[str] = PARAMETERS) -> ParamDict:
        for param in parameters:
            assert param in PARAMETERS, f"Invalid parameter: {param}"

        params: ParamDict = {
            "few_shot_enabled": False,
        }
        defaults = self.defaults()

        if "rag_mode" in parameters:
            params["rag_mode"] = trial.suggest_categorical("rag_mode", self.rag_modes)
        else:
            params["rag_mode"] = defaults["rag_mode"]

        if "template_name" in parameters:
            params["template_name"] = trial.suggest_categorical(
                "template_name", self.template_names
            )
        else:
            params["template_name"] = defaults["template_name"]

        if "response_synthesizer_llm" in parameters:
            params["response_synthesizer_llm"] = trial.suggest_categorical(
                "response_synthesizer_llm", self.response_synthesizer_llms
            )
        else:
            params["response_synthesizer_llm"] = defaults["response_synthesizer_llm"]


        if "rag_retriever" in parameters:
            params.update(**self.rag_retriever.sample(trial))
        else:
            params.update(**self.rag_retriever.defaults())

        if "splitter" in parameters:
            params.update(**self.splitter.sample(trial))
        else:
            params.update(**self.splitter.defaults())

        if "reranker" in parameters:
            params["reranker_enabled"] = trial.suggest_categorical(
                "reranker_enabled", self.reranker_enabled
            )
            if params["reranker_enabled"]:
                params.update(**self.reranker.sample(trial))
        else:
            params["reranker_enabled"] = False

          

        if params["rag_mode"] == "sub_question_rag":
            if "sub_question_rag" in parameters:
                params.update(**self.sub_question_rag.sample(trial))
            else:
                params.update(**self.sub_question_rag.defaults())
  


        return params

    def get_cardinality(self) -> int:
        card = 0

        sub_card = (
            len(self.template_names)
            * len(self.response_synthesizer_llms)
            * len(self.reranker_enabled)
        )
    
        sub_card *= self.rag_retriever.get_cardinality()
        sub_card *= self.splitter.get_cardinality()
        if True in self.reranker_enabled:
            sub_card *= self.reranker.get_cardinality()

        card += sub_card

        return card
