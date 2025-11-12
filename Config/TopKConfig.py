from pydantic import BaseModel, Field
from Config.SearchSpaceMix import *
import typing as T

class TopK(BaseModel, SearchSpaceMixin):
    kmin: int = Field(
        default=2, description="Minimum value for number of items to retrieve."
    )
    kmax: int = Field(
        default=20, description="Maximum value for number of items to retrieve."
    )
    log: bool = Field(
        default=False,
        description="Whether to use a logarithmic scale instead of linear for top_k.",
    )
    step: int = Field(default=1, description="Step size for top_k.")

    def defaults(self, prefix: str = "") -> T.Dict[str, T.Any]:
        return {
            f"{prefix}top_k": 5,
        }

    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        name = f"{prefix}top_k"
        return {
            name: IntDistribution(self.kmin, self.kmax, log=self.log, step=self.step)
        }

    def get_cardinality(self) -> int:
        return get_dist_cardinality(self.kmin, self.kmax, self.step)