from abc import abstractmethod
from typing import Any
from Tuner.BasicTuner import BasicTuner

class BasicBOTuner(BasicTuner):
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