from pydantic import BaseModel, Field
import typing as T
from datetime import timedelta

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
