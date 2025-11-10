
import math
import os
import typing as T
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path

# 🔧 修复CUDA设备冲突：正确处理CUDA_VISIBLE_DEVICES映射
# 当设置CUDA_VISIBLE_DEVICES时，PyTorch会将可见设备重新编号为0,1,2...
# 所以我们应该使用逻辑设备号0，而不是物理设备号
DEVICE_ID = 0# if os.environ.get('CUDA_VISIBLE_DEVICES') else 0

import pandas as pd
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
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings, f
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from hammer.configuration import NDIGITS, cfg
from hammer.helpers import (
    get_max_float,
    get_max_int,
    get_min_float,
    get_min_int,
    get_unique_bools,
    get_unique_strings,
)
from hammer.llm import LLMs
from hammer.storage import (
    CragTask3HF,
    DRDocsHF,
    FinanceBenchHF,
    HotPotQAHF,
    PartitionMap,
    HammerQADataset,
    TwoWikiMultiHopQA,      # 🔥 新增
    UnifiedJSONDataset,     # 🔥 新增统一数据集支持
)

ParamDict = T.Dict[str, str | int | float | bool]
# from hammer.tuner.main_tuner_mcts import DEVICE_ID



# class Splitter(BaseModel, SearchSpaceMixin):
 

LOCAL_EMBEDDING_MODELS = (
    [model.model_name for model in cfg.local_models.embedding]
    if cfg.local_models.embedding
    else []
)

DEFAULT_EMBEDDING_MODELS: T.List[str] = list(
    set(
        [
            "BAAI/bge-small-en-v1.5",  # first embedding model is the default
            "BAAI/bge-large-en-v1.5",
            "thenlper/gte-large",
            "mixedbread-ai/mxbai-embed-large-v1",
            "WhereIsAI/UAE-Large-V1",
            "avsolatorio/GIST-large-Embedding-v0",
            "w601sxs/b1ade-embed",
            "Labib11/MUG-B-1.6",
            "sentence-transformers/all-MiniLM-L12-v2",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "BAAI/bge-base-en-v1.5",
            "FinLang/finance-embeddings-investopedia",
            "baconnier/Finance2_embedding_small_en-V1.5",
            # "thomaskim1130/stella_en_400M_v5-FinanceRAG-v2",
            # "Alibaba-NLP/gte-large-en-v1.5",
            # "Alibaba-NLP/gte-base-en-v1.5",
            # "llmrails/ember-v1",
            # "jamesgpt1/sf_model_e5",
            # "mixedbread-ai/mxbai-embed-2d-large-v1",
            # "intfloat/e5-large-v2",
        ]
        + LOCAL_EMBEDDING_MODELS
    )
)

ALL_LLMS = list(LLMs.keys())

LOCAL_LLMS = (
    [model.model_name for model in cfg.local_models.generative]
    if cfg.local_models.generative
    else []
)

DEFAULT_LLMS: T.List[str] = list(
    set[str](
        [
            "gpt-4o-mini",      # gaochao API
            "Qwen2_5-7b",       # 硅基流动API (Qwen2.5-7B)
            "Qwen2-7b",         # 硅基流动API (Qwen2-7B) 🔥 新增支持
            "DeepSeek-R1-32b",  # 硅基流动API (DeepSeek-R1)
            "Qwen2.5-72b",      # 硅基流动API (Qwen2.5-72B)
        ]
        + LOCAL_LLMS
    )
)
assert set(DEFAULT_LLMS).issubset(set(ALL_LLMS))

RESPONSE_SYNTHESIZER_LLMS: T.List[str] = [
    "gpt-4o-mini",      # gaochao API
    "Qwen2_5-7b",       # 硅基流动API (Qwen2.5-7B)
    "Qwen2-7b",         # 硅基流动API (Qwen2-7B) 🔥 新增支持
    "DeepSeek-R1-32b",  # 硅基流动API (DeepSeek-R1)
    "Qwen2.5-72b",      # 硅基流动API (Qwen2.5-72B)
] + LOCAL_LLMS
assert set(RESPONSE_SYNTHESIZER_LLMS).issubset(set(ALL_LLMS))

FUNCTION_CALLING_LLMS: T.List[str] = [
    "gpt-4o-mini",      # gaochao API支持函数调用
    "Qwen2_5-7b",       # 硅基流动API支持函数调用 (Qwen2.5-7B)
    "Qwen2-7b",         # 硅基流动API支持函数调用 (Qwen2-7B) 🔥 新增支持
    "DeepSeek-R1-32b",  # 硅基流动API支持函数调用 (DeepSeek-R1)
    "Qwen2.5-72b",      # 硅基流动API支持函数调用 (Qwen2.5-72B)
] + LOCAL_LLMS
assert set(FUNCTION_CALLING_LLMS).issubset(set(ALL_LLMS))

CHEAP_LLMS: T.List[str] = [
    "gpt-4o-mini",  # gaochao API价格较低
    "Qwen2_5-7b",   # 硅基流动API价格较低 (Qwen2.5-7B)
    "Qwen2-7b",     # 硅基流动API价格较低 (Qwen2-7B) 🔥 新增支持
] + LOCAL_LLMS
assert set(CHEAP_LLMS).issubset(set(ALL_LLMS))
assert set(CHEAP_LLMS).issubset(set(ALL_LLMS))

NON_REASONING_LLMS: T.List[str] = list(
    set(
        [
            "gpt-4o-mini",      # gaochao API非推理模型
            "Qwen2_5-7b",       # 硅基流动API非推理模型 (Qwen2.5-7B)
            "Qwen2-7b",         # 硅基流动API非推理模型 (Qwen2-7B) 🔥 新增支持
            "DeepSeek-R1-32b",  # 硅基流动API非推理模型 (DeepSeek-R1)
            "Qwen2.5-72b",      # 硅基流动API非推理模型 (Qwen2.5-72B)
        ]
        + LOCAL_LLMS
    )
)
assert set(NON_REASONING_LLMS).issubset(set(ALL_LLMS))

RAG_MODES: T.List[str] = [
    "rag",  #  first mode is the default
    "react_rag_agent",
    "critique_rag_agent",
    "sub_question_rag",
    "lats_rag_agent",
    "no_rag",
]

TEMPLATE_NAMES = [
    "default",  # first template is the default
    "concise",
    "CoT",
    "finance-expert",
]

PARAMETERS = [
    "rag_retriever",
    "splitter",
    "additional_context",
    "few_shot_retriever",
    "hyde",
    "reranker",
    "rag_mode",
    "sub_question_rag",
    "critique_rag_agent",
    "lats_rag_agent",
    "react_rag_agent",
    "response_synthesizer_llm",
    "template_name",
]













class RetrieverSearchSpace(BaseModel):
    """Search space over retrievers."""

    model_config = ConfigDict(extra="forbid")  # Forbids unknown fields
    rag_modes: T.List[str] = Field(
        default_factory=lambda: ["rag"],
        description='List of RAG modes, restricted to "rag" for this specific search space.',
    )
    non_search_space_params: T.List[str] = Field(
        default_factory=lambda: ["enforce_full_evaluation"],
        description="Parameters not part of the hyperparameter search space.",
    )
    response_synthesizer_llms: T.List[str] = Field(
        default_factory=lambda: DEFAULT_LLMS,
        description="LLMs used for response synthesis.",
    )
    rag_retriever: Retriever = Field(
        default_factory=lambda: Retriever(top_k=TopK(kmax=128, log=True)),
        description="Configuration for the RAG retriever.",
    )
    splitter: Splitter = Field(
        default_factory=Splitter, description="Configuration for the text splitter."
    )
    hyde_enabled: T.List[bool] = Field(
        default_factory=lambda: [True, False], description="Whether HyDE is enabled."
    )
    hyde: Hyde = Field(default_factory=Hyde, description="Configuration for HyDE.")
    additional_context_enabled: T.List[bool] = Field(
        default_factory=lambda: [True, False],
        description="Whether additional context is enabled.",
    )
    additional_context: AdditionalContext = Field(
        default_factory=AdditionalContext,
        description="Configuration for additional context.",
    )

    def defaults(self) -> ParamDict:
        return {
            "rag_mode": self.rag_modes[0],
            "response_synthesizer_llm": self.response_synthesizer_llms[0],
            "additional_context_enabled": False,
            "hyde_enabled": False,
            **self.rag_retriever.defaults(),
            **self.splitter.defaults(),
        }

    def build_distributions(
        self, params: T.Dict[str, T.Any] | T.List[str] | None = None
    ) -> T.Dict[str, BaseDistribution]:
        distributions: dict[str, BaseDistribution] = {
            "rag_mode": CategoricalDistribution(self.rag_modes),
            "response_synthesizer_llm": CategoricalDistribution(
                self.response_synthesizer_llms
            ),
            "hyde_enabled": CategoricalDistribution(self.hyde_enabled),
            "additional_context_enabled": CategoricalDistribution(
                self.additional_context_enabled
            ),
            **self.rag_retriever.build_distributions(prefix="rag_"),
            **self.splitter.build_distributions(),
        }
        if True in self.hyde_enabled:
            distributions.update(self.hyde.build_distributions())
        if True in self.additional_context_enabled:
            distributions.update(self.additional_context.build_distributions())

        if params is not None:
            reduced_distributions = {
                key: val for key, val in distributions.items() if key in params
            }
            return reduced_distributions

        return distributions

    def sample(self, trial: Trial, prefix: str = "") -> ParamDict:
        params: ParamDict = {
            "rag_mode": trial.suggest_categorical("rag_mode", self.rag_modes),
            "response_synthesizer_llm": trial.suggest_categorical(
                "response_synthesizer_llm", self.response_synthesizer_llms
            ),
            "hyde_enabled": trial.suggest_categorical(
                "hyde_enabled", self.hyde_enabled
            ),
            "additional_context_enabled": trial.suggest_categorical(
                "additional_context_enabled", self.additional_context_enabled
            ),
            **self.rag_retriever.sample(trial, prefix="rag_"),
            **self.splitter.sample(trial),
        }
        if params["hyde_enabled"]:
            params.update(**self.hyde.sample(trial))
        if params["additional_context_enabled"]:
            params.update(**self.additional_context.sample(trial))
        return params

    def get_cardinality(self) -> int:
        return (
            self.rag_retriever.get_cardinality()
            * self.splitter.get_cardinality()
            * self.hyde.get_cardinality()
            * self.additional_context.get_cardinality()
            * len(self.hyde_enabled)
            * len(self.additional_context_enabled)
        )

class AgentSearchSpace(BaseModel):
    model_config = ConfigDict(extra="forbid")  # Forbids unknown fields
    llms: T.List[str] = [
        "gpt-4o-mini",
    ]
    rag_modes: T.List[str] = [
        "no_rag",
        "rag",
    ]
    prompt_names: T.List[str] = [
        "default",
        "concise",
        "aggressive",
    ]
    embedding_models: T.List[str] = [
        "BAAI/bge-small-en-v1.5",
        "sentence-transformers/all-MiniLM-L12-v2",
    ]
    splitter: Splitter = Splitter()

EmbeddingDeviceType = T.Literal["onnx-cpu", "cpu", "mps", "cuda", None]

class Block(BaseModel):
    name: str = Field(default="global", description="Block name")
    num_trials: int = Field(default=1000, description="Number of trials.")
    components: T.List[str] = Field(
        default_factory=lambda: PARAMETERS, description="Block components"
    )

class OptimizationConfig(BaseModel):
    method: T.Literal["expanding", "knee"] = Field(
        default="expanding",
        description="Method for optimization, e.g., expanding window or knee point detection.",
    )
    blocks: T.List[Block] = Field(
        default_factory=lambda: [Block()], description="List of optimization blocks."
    )
    shuffle_blocks: bool = Field(
        default=False,
        description="Whether to shuffle the order of optimization blocks.",
    )
    num_trials: int = Field(
        default=1000, description="Total number of optimization trials."
    )
    model_config = ConfigDict(extra="forbid")  # Forbids unknown fields
    baselines: T.List[T.Dict[str, T.Any]] = Field(
        default_factory=list,
        description="List of baseline configurations to compare against.",
    )
    shuffle_baselines: bool = Field(
        default=True, description="Whether to shuffle the order of baselines."
    )
    cpus_per_trial: int = Field(
        default=2, description="Number of CPUs allocated per trial."
    )
    gpus_per_trial: int | float = Field(
        default=0.0, description="Number of GPUs allocated per trial."
    )
    embedding_device: EmbeddingDeviceType = Field(
        default=f"cuda:{DEVICE_ID}",  # 🔧 修复：使用字符串"cuda"而不是整数DEVICE_ID
        description="Device to use for embeddings (e.g., 'cpu', 'cuda', 'onnx-cpu'). Use `None` to auto-detect.",
    )
    use_hf_embedding_models: bool = Field(
        default=False, description="Whether to use HuggingFace embedding models."
    )
    raise_on_failed_trial: bool = Field(
        default=False, description="Whether to raise an exception if a trial fails."
    )
    max_concurrent_trials: int = Field(
        default=10, description="Maximum number of trials to run concurrently."
    )
    num_eval_samples: int = Field(
        default=500, description="Number of samples to use for evaluation."
    )
    num_eval_batch: int = Field(default=5, description="Batch size for evaluation.")
    max_eval_failure_rate: float = Field(
        default=0.5, description="Maximum allowed failure rate during evaluation."
    )
    max_trial_cost: float = Field(
        default=10.00, description="Maximum allowed cost per trial."
    )
    num_random_trials: int = Field(
        default=100, description="Number of random trials to run initially."
    )
    num_retries_unique_params: int = Field(
        default=100,
        description="Number of retries to find unique parameters for a trial.",
    )
    num_prompt_optimization_batch: int = Field(
        default=50, description="Batch size for prompt optimization."
    )
    rate_limiter_max_coros: int = Field(
        default=3, description="Maximum number of coroutines for the rate limiter."
    )
    rate_limiter_period: int = Field(
        default=10, description="Period in seconds for the rate limiter."
    )
    skip_existing: bool = Field(
        default=True, description="Whether to skip trials with existing results."
    )
    num_warmup_steps_timeout: int = Field(
        default=3, description="Number of warmup steps for timeout pruner."
    )
    num_warmup_steps_costout: int = Field(
        default=2, description="Number of warmup steps for cost pruner."
    )
    num_warmup_steps_pareto: int = Field(
        default=30, description="Number of warmup steps for Pareto pruner."
    )
    use_pareto_pruner: bool = Field(
        default=True, description="Whether to use the Pareto pruner."
    )
    use_cost_pruner: bool = Field(
        default=True, description="Whether to use the cost pruner."
    )
    use_runtime_pruner: bool = Field(
        default=True, description="Whether to use the runtime pruner."
    )
    pareto_pruner_success_rate: float = Field(
        default=0.9, description="Success rate threshold for Pareto pruner."
    )
    pareto_eval_success_rate: float = Field(
        default=0.9, description="Success rate threshold for Pareto evaluation."
    )
    raise_on_invalid_baseline: bool = Field(
        default=False,
        description="Whether to raise an exception for invalid baselines.",
    )
    baselines_cycle_llms: bool = Field(
        default=False, description="Whether to cycle through LLMs for baselines."
    )
    use_toy_baselines: bool = Field(
        default=False, description="Whether to use toy baselines."
    )
    use_individual_baselines: bool = Field(
        default=True, description="Whether to use individual component baselines."
    )
    use_agent_baselines: bool = Field(
        default=True, description="Whether to use agent-specific baselines."
    )
    use_variations_of_baselines: bool = Field(
        default=True, description="Whether to use variations of baselines."
    )
    use_pareto_baselines: bool = Field(
        default=False,
        description="Whether to use baselines from the Pareto front, switch to True for transfer learning",
    )
    objective_1_name: T.Literal["accuracy", "retriever_recall", "joint_f1", "answer_f1", "f1"] = Field(
        default="accuracy", description="Name of the first optimization objective."
    )
    objective_2_name: T.Optional[T.Literal[
        "p80_time", "llm_cost_mean", "retriever_context_length"
    ]] = Field(
        default="llm_cost_mean",
        description="Name of the second optimization objective (optional for single-objective optimization).",
    )
    obj1_zscore: float = Field(
        default=1.645,
        description="Z-score for the first objective (e.g., for confidence interval).",
    )
    obj2_zscore: float = Field(
        default=1.645, description="Z-score for the second objective."
    )
    # 修改这一行：添加 "mcts" 支持
    sampler: T.Literal["tpe", "hierarchical", "mcts"] = Field(
        default="tpe",
        description='Type of sampler to use (e.g., "tpe", "hierarchical", "mcts").',
    )
    
    # 添加 MCTS 特定的配置字段
    mcts_iteration_limit: int = Field(
        default=100,
        description="Number of iterations for MCTS search per parameter suggestion.",
    )
    mcts_exploration_constant: float = Field(
        default=1.414,  # sqrt(2)
        description="UCB exploration constant for MCTS (higher values encourage more exploration).",
    )
    mcts_max_depth: int = Field(
        default=10,
        description="Maximum search depth for MCTS tree traversal.",
    )
    
    # MCTS 批次化训练配置
    enable_batch_training: bool = Field(
        default=False,
        description="Enable batch training mode for MCTS (split training data into batches).",
    )
    train_data_size: int = Field(
        default=200,
        description="Number of training examples to use for batch training.",
    )
    num_batches: int = Field(
        default=4,
        description="Number of batches to split training data into.",
    )
    eval_data_size: int = Field(
        default=50,
        description="Number of examples to use for real evaluation after each batch.",
    )
    
    ############################
    # seeder_timeout settings
    # --------------------------
    # 1 hour: 3600
    # 1 day: 86400
    # no wait: 0
    # wait until finished: None
    # -------------------------
    # main optimization starts in parallel after timeout
    # while seeding continues
    seeder_timeout: float | None = Field(
        default=3600,
        description="Timeout in seconds for the seeder process. None means wait indefinitely.",
    )

class TransferLearningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")  # Forbids unknown fields
    studies: T.List[str] = Field(
        default_factory=list,
        description="List of study names to use for transfer learning.",
    )
    max_fronts: int = Field(
        default=2,
        description="Maximum number of Pareto fronts to consider from previous studies.",
    )
    max_total: int = Field(
        default=100, description="Maximum total number of configurations to transfer."
    )
    success_rate: float = Field(
        default=0.9, description="Minimum success rate for transferred configurations."
    )
    embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Embedding model used for comparing configurations in transfer learning.",
    )

class TimeoutConfig(BaseModel):
    embedding_timeout_active: bool = Field(
        default=False, description="Whether embedding timeout is active."
    )
    embedding_max_time: int = Field(
        default=3600 * 4, description="Maximum time in seconds for embeddings."
    )
    embedding_min_chunks_to_process: int = Field(
        default=100,
        description="Minimum number of chunks to process before embedding timeout.",
    )
    embedding_min_time_to_process: int = Field(
        default=120,
        description="Minimum time in seconds to process before embedding timeout.",
    )
    eval_timeout: int = Field(
        default=3600 * 10,
        description="Maximum time in seconds for the entire evaluation process.",
    )
    single_eval_timeout: int = Field(
        default=3600 * 2,
        description="Maximum time in seconds for a single evaluation run.",
    )
    onnx_timeout: int = Field(
        default=600, description="Maximum time in seconds for ONNX model operations."
    )

class Evaluation(BaseModel):
    mode: T.Literal["single", "random", "consensus", "retriever"] = Field(
        default="single", description="Evaluation mode."
    )
    llms: T.List[str] = Field(
        default_factory=lambda: ["gpt-4o-mini"],
        description="List of LLMs to use for evaluation. If 'single' mode is chosen, the first list item will be used.",
    )
    raise_on_exception: bool = Field(
        default=False,
        description="Whether to raise an exception if an error occurs during evaluation.",
    )
    use_tracing_metrics: bool = Field(
        default=False, description="Whether to use tracing metrics during evaluation."
    )
    min_reporting_success_rate: float = Field(
        default=0.5,
        description="Minimum success rate for reporting evaluation results.",
    )

class ParetoConfig(BaseModel):
    """
    Parameters that are used to override the study config for the Pareto front evaluation,
    for instance, `optimization__skip_existing` is used to override `optimization.skip_existing`.
    """

    name: str = Field(description="Name of the Pareto configuration/study.")
    raise_on_same_study: bool = Field(
        default=True,
        description="Whether to raise an error if the Pareto study name is the same as the main study.",
    )
    reuse_study: bool = Field(
        default=False, description="Whether to reuse an existing Pareto study."
    )
    optimization__skip_existing: bool = Field(
        default=True,
        description="Override for optimization.skip_existing for Pareto evaluation, switch to false when using same study.",
    )
    optimization__use_pareto_pruner: bool = Field(
        default=False,
        description="Override for optimization.use_pareto_pruner for Pareto evaluation.",
    )
    optimization__use_cost_pruner: bool = Field(
        default=False,
        description="Override for optimization.use_cost_pruner for Pareto evaluation.",
    )
    optimization__use_runtime_pruner: bool = Field(
        default=False,
        description="Override for optimization.use_runtime_pruner for Pareto evaluation.",
    )
    optimization__num_eval_samples: int = Field(
        description="Override for optimization.num_eval_samples for Pareto evaluation."
    )  # No default
    replacement_llm_name: str = Field(
        default="",
        description="LLM name to replace in configurations for Pareto evaluation (e.g., to test a new LLM on existing good configurations).",
    )
    dataset__partition_map: PartitionMap = Field(
        default_factory=lambda: PartitionMap(
            sample="sample",
            train="test",
            test="holdout",
            holdout="holdout",
        ),
        description="Override for dataset partition mapping for Pareto evaluation.",
    )

class StudyConfig(BaseSettings):
    name: str = Field(description="Name of the Optuna study.")
    dataset: T.Annotated[  # type: ignore
        T.Union[
            *HammerQADataset.__subclasses__(),  # type: ignore
            *HotPotQAHF.__subclasses__(),  # type: ignore
            *FinanceBenchHF.__subclasses__(),  # type: ignore
            *CragTask3HF.__subclasses__(),  # type: ignore
            *DRDocsHF.__subclasses__(),  # type: ignore
        ],
        Field(discriminator="xname"),
    ] = Field(description="Dataset configuration.")
    evaluation: Evaluation = Field(
        default_factory=Evaluation, description="LLM-as-a-judge configuration."
    )
    reuse_study: bool = Field(
        default=True, description="Whether to reuse an existing study."
    )
    recreate_study: bool = Field(
        default=True,
        description="Whether to recreate the study if it already exists (potentially deleting old data).",
    )
    search_space: SearchSpace = Field(
        default_factory=SearchSpace,
        description="Search space configuration for the optimization.",
    )
    optimization: OptimizationConfig = Field(
        default_factory=OptimizationConfig,
        description="Optimization process configuration.",
    )
    pareto: T.Optional[ParetoConfig] = Field(
        default=None, description="Optional configuration for Pareto front evaluation."
    )
    transfer_learning: TransferLearningConfig = Field(
        default_factory=TransferLearningConfig,
        description="Transfer learning configuration.",
    )
    timeouts: TimeoutConfig = Field(
        default_factory=TimeoutConfig,
        description="Timeout configurations for various stages.",
    )
    toy_mode: bool = Field(
        default=False, description="Whether to run in toy mode (with smaller dataset)."
    )

    model_config = SettingsConfigDict(
        extra="forbid",  # Forbids unknown fields
        yaml_file=cfg.study_config_file or Path("Idontexist"),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: T.Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> T.Tuple[PydanticBaseSettingsSource, ...]:
        """Study config can be loaded from a yaml file.

        Use HAMMER_STUDY_CONFIG_FILE env var or
        'study_config_file: <path> in the top-level of config.yaml
        to choose a study config file, or use the from_file factory method.

        Parameters passed to StudyConfig.__init__ will take precedence
        over the yaml file.
        """
        if cfg.study_config_file and not cfg.study_config_file.exists():
            raise ValueError(
                f"Study configuration file cannot be found at {cfg.study_config_file.resolve()}"
            )

        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    @classmethod
    def from_file(cls, path: Path | str, *args, **kwargs) -> "StudyConfig":
        """Use from_file to load from a given config file path.

        *args and **kwargs are the same as the StudyConfig constructor
        and take precedence over values loaded from the config file.

        cfg.study_config_file is ignored when this method is used.
        """
        if not Path(path).exists():
            raise ValueError(
                f"Study configuration file cannot be found at {Path(path).resolve()}"
            )

        klass = deepcopy(cls)
        _orig = klass.model_config.pop("yaml_file", None)
        klass.model_config = SettingsConfigDict(**cls.model_config, yaml_file=path)
        instance = klass(*args, **kwargs)
        klass.model_config["yaml_file"] = _orig
        return instance

    def replace_llm_name(self, params: T.Dict[str, T.Any]):
        """
        Replace the LLM name in the params with the replacement_llm_name.
        With this functionality, we can easily run historical flows with a different LLM.
        """
        assert self.pareto, "No Pareto config is set"
        assert self.pareto.replacement_llm_name, "No replacement LLM name is set"

        replacement_llm_name = self.pareto.replacement_llm_name
        params["response_synthesizer_llm"] = replacement_llm_name

    @property
    def is_retriever_study(self) -> bool:
        return isinstance(self.search_space, RetrieverSearchSpace)


def get_default_study_name():
    if os.path.exists("studies/private.yaml"):
        return "studies/private.yaml"
    return "studies/hotpot-toy.yaml"
