import typing as T
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings
)
from Config.SearchSpace import SearchSpace
from Config.OptimizationConfig import OptimizationConfig
from Config.TimeoutConfig import TimeoutConfig


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


class TunerConfig(BaseSettings):
    name: str = Field(
        default="default", description="Name of the tuner."
    )
    tuner_params: T.List[str] = Field(
        default_factory=lambda: ["template_name", "response_synthesizer_llm", "reranker", "rag_retriever", "reranker", "faiss", "sub_question"],
        description="Parameters to tune."
    )
    evaluation: Evaluation = Field(
        default_factory=Evaluation, description="LLM-as-a-judge configuration."
    )
    reuse_study: bool = Field(
        default=False, description="Whether to reuse an existing study."
    )
    recreate_study: bool = Field(
        default=False,
        description="Whether to recreate the study if it already exists (potentially deleting old data).",
    )
    search_space: SearchSpace = Field(
        default_factory=SearchSpace,
        description="Search space configuration for the optimization.",
    )
    cluster_round_sync: bool = Field(
        default=False,
        description="Enable round-synchronous cluster-shared LGBO execution.",
    )
    cluster_k: int = Field(
        default=3,
        description="Number of query clusters for cluster-round LGBO.",
    )
    cluster_random_seed: int = Field(
        default=42,
        description="Random seed used by query clustering in cluster-round LGBO.",
    )
    optimization: OptimizationConfig = Field(
        default_factory=OptimizationConfig,
        description="Optimization process configuration.",
    )

    timeouts: TimeoutConfig = Field(
        default_factory=TimeoutConfig,
        description="Timeout configurations for various stages.",
    )
    
