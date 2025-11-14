from pydantic import BaseModel, ConfigDict, Field
import typing as T
from Common.Constants import TEMPLATE_NAMES, DEFAULT_LLMS
from Config.RetrieverConfig import Retriever
from Config.TopKConfig import TopK
from Config.RerankConfig import Reranker
from Config.SearchSpaceMix import *
from Config.QueryConfig import QueryConfig


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

    sub_question: QueryConfig = Field(
        default_factory=QueryConfig, description="Configuration for the sub-question."
    )



    def _defaults(self) -> ParamDict:
        return {
            "template_name": self.template_names[0],
            "response_synthesizer_llm": self.response_synthesizer_llms[0],
            "reranker_enabled": False,
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
            "reranker_enabled": CategoricalDistribution(self.reranker_enabled),
        }

        distributions.update(self.rag_retriever.build_distributions())
        if True in self.reranker_enabled:
            distributions.update(self.reranker.build_distributions())


        if params is not None:
            reduced_distributions = {
                key: val for key, val in distributions.items() if key in params
            }
            return reduced_distributions

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


        if "rag_retriever" in parameters:
            params.update(**self.rag_retriever.sample(trial))
        else:
            params.update(**self.rag_retriever.defaults())

        if "reranker" in parameters:
                params.update(**self.reranker.sample(trial))
                params["reranker_enabled"] = True
        else:
            params["reranker_enabled"] = False


        if "sub_question" in parameters:
            params.update(**self.sub_question.sample(trial))
        else:
            params.update(**self.sub_question.defaults())
        return params

    def get_cardinality(self) -> int:
        card = 0

        sub_card = (
            len(self.template_names)
            * len(self.response_synthesizer_llms)
            * len(self.reranker_enabled)
        )
    
        sub_card *= self.rag_retriever.get_cardinality()
        if True in self.reranker_enabled:
            sub_card *= self.reranker.get_cardinality()

        card += sub_card

        return card
