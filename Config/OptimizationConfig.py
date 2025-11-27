from pydantic import BaseModel, ConfigDict, Field
import typing as T
from Common.Constants import PARAMETERS


class Block(BaseModel):
    name: str = Field(default="global", description="Block name")
    num_trials: int = Field(default=1000, description="Number of trials.")
    components: T.List[str] = Field(
        default_factory=lambda: list(PARAMETERS), description="Block components"
    )


class OptimizationConfig(BaseModel):
    method: T.Literal["expanding", "knee"] = Field(
        default="expanding",
        description="Method for optimization, e.g., expanding window or knee point detection.",
    )
    blocks: T.List[Block] = Field(
        default_factory=lambda: [Block()], description="List of optimization blocks."
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
    embedding_device: T.Union[str, T.List[str], None] = Field(
        default=["cuda:0"],
        description="Device to use for embeddings in Hugging Face format (e.g., 'cuda:0', 'cpu', ['cuda:0', 'cuda:1']). Use `None` to auto-detect.",
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

    objective_1_name: T.Literal["accuracy", "retriever_recall"] = Field(
        default="accuracy", description="Name of the first optimization objective."
    )
    objective_2_name: T.Literal[
        "p80_time", "llm_cost_mean", "retriever_context_length"
    ] = Field(
        default="llm_cost_mean",
        description="Name of the second optimization objective.",
    )
    obj1_zscore: float = Field(
        default=1.645,
        description="Z-score for the first objective (e.g., for confidence interval).",
    )
    obj2_zscore: float = Field(
        default=1.645, description="Z-score for the second objective."
    )
    sampler: T.Literal["tpe", "hierarchical", "llmbo"] = Field(
        default="tpe",
        description='Type of sampler to use (e.g., "tpe", "hierarchical").',
    )