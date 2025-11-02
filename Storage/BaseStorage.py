from dataclasses import dataclass, field
from typing import (
    Any,
    Optional

)
from Storage.NameSpace import Namespace


@dataclass
class BaseStorage:
    config: Optional[Any] = field(default=None)
    namespace: Optional[Namespace] = field(default=None)
