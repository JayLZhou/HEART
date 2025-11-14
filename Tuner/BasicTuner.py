from abc import ABC, abstractmethod
from typing import Any

class BasicTuner(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _create_tuner(self):
        pass

    @abstractmethod
    def get_sampler(self):
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)

    @abstractmethod
    def save_config(self, *args: Any, **kwds: Any) -> Any:
        pass