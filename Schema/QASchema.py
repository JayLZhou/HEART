from pydantic import BaseModel
import typing as T

class QAPair(BaseModel):
    question: str
    answer: str
    id: str
    dataset_name: str = ""  # Dataset name identifier
    metadata: T.Dict[str, T.Any] = {}  # Additional metadata