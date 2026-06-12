import typing as T

from optuna.distributions import BaseDistribution, CategoricalDistribution, IntDistribution
from pydantic import BaseModel, Field

from Config.SearchSpaceMix import ParamDict, SearchSpaceMixin


class FaissSearchSpace(BaseModel, SearchSpaceMixin):
    hnsw_m_min: int = Field(default=16, description="Minimum HNSW M value.")
    hnsw_m_max: int = Field(default=64, description="Maximum HNSW M value.")
    hnsw_m_step: int = Field(default=8, description="Step size for HNSW M.")

    hnsw_ef_search_min: int = Field(default=32, description="Minimum HNSW efSearch value.")
    hnsw_ef_search_max: int = Field(default=256, description="Maximum HNSW efSearch value.")
    hnsw_ef_search_step: int = Field(default=32, description="Step size for HNSW efSearch.")

    hnsw_ef_construction_min: int = Field(default=40, description="Minimum HNSW efConstruction value.")
    hnsw_ef_construction_max: int = Field(default=320, description="Maximum HNSW efConstruction value.")
    hnsw_ef_construction_step: int = Field(default=40, description="Step size for HNSW efConstruction.")

    metrics: T.List[str] = Field(
        default_factory=lambda: ["l2", "inner_product"],
        description="Supported FAISS distance metrics for HNSWFlat.",
    )

    def defaults(self, prefix: str = "") -> ParamDict:
        return {
            f"{prefix}faiss_hnsw_m": 32,
            f"{prefix}faiss_hnsw_ef_search": 64,
            f"{prefix}faiss_hnsw_ef_construction": 40,
            f"{prefix}faiss_metric": self.metrics[0],
        }

    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        return {
            f"{prefix}faiss_hnsw_m": IntDistribution(
                low=self.hnsw_m_min,
                high=self.hnsw_m_max,
                step=self.hnsw_m_step,
            ),
            f"{prefix}faiss_hnsw_ef_search": IntDistribution(
                low=self.hnsw_ef_search_min,
                high=self.hnsw_ef_search_max,
                step=self.hnsw_ef_search_step,
            ),
            f"{prefix}faiss_hnsw_ef_construction": IntDistribution(
                low=self.hnsw_ef_construction_min,
                high=self.hnsw_ef_construction_max,
                step=self.hnsw_ef_construction_step,
            ),
            f"{prefix}faiss_metric": CategoricalDistribution(self.metrics),
        }

    def get_cardinality(self) -> int:
        hnsw_m_choices = ((self.hnsw_m_max - self.hnsw_m_min) // self.hnsw_m_step) + 1
        ef_search_choices = ((self.hnsw_ef_search_max - self.hnsw_ef_search_min) // self.hnsw_ef_search_step) + 1
        ef_construction_choices = (
            (self.hnsw_ef_construction_max - self.hnsw_ef_construction_min) // self.hnsw_ef_construction_step
        ) + 1
        return hnsw_m_choices * ef_search_choices * ef_construction_choices * len(self.metrics)
