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
from Common.Constants import NDIGITS, TEMPLATE_NAMES, DEFAULT_LLMS
ParamDict = T.Dict[str, str | int | float | bool]
# from Common.Utils import get_dist_cardinality

def get_dist_cardinality(min: int | float, max: int | float, step: int | float) -> int:
    """Returns the cardinality of an integer or float distribution"""
    assert min <= max
    assert step > 0
    return int((max - min) / step) + 1
    
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


