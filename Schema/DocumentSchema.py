import json
from typing import List,Optional,Dict
import os, requests
from llama_index.core.schema import NodeWithScore
import csv
from dataclasses import dataclass, asdict

@dataclass
class Context:
    def __init__(self, text: str, id: int, score: float = 0.0) -> None:
        self.text: str = text
        self.id: int = id
        self.score: float = score
    @property
    def as_dict(self):
        return asdict(self)
class Document:
    """
    Represents a document consisting of a question, answers, and contexts.

    Attributes:
        question (Question): The question associated with the document.
        answers (Answer): The answers to the question.
        contexts (list[Context]): A list of related contexts.
        reorder_contexts (list[Context] or None): A reordered list of contexts based on relevance.
    """
    def __init__(self, question: str, contexts: list = None , id: int = None) -> None:
        """
        Initializes a Document instance.

        Args:
            question (Question): The question associated with the document.
            contexts (list[Context], optional): A list of contexts related to the question.

        """
        if isinstance(contexts[0], NodeWithScore):
            contexts = [Context(text=node.text, id=i, score=node.score) for i, node in enumerate(contexts)]
        self.question: str = question
        self.contexts: List[Context] = contexts
        self.reorder_contexts: List[Context] = None
        self.id = str(id) 



    def to_dict(self) -> Dict[str, Optional[object]]:
        """
        Converts the document into a dictionary representation.

        Returns:
            dict: A dictionary containing the question, answers, and contexts.
        """
        return {
            "question": self.question,
            "contexts": [ctx.to_dict() for ctx in self.contexts]
        }
    def to_dict_reoreder(self) -> Dict[str,Optional[object]]:
        return {
            "question" : self.question.question,
            "answers" : self.answers.answers,
            "contexts" : [ctx.to_dict() for ctx in self.reorder_contexts]
        }
    def __str__(self) -> str:
        """
        Returns a string representation of the Document instance.

        Returns:
            str: The formatted document information.

        Example:
            ```python
            d = Document(Question("What is the capital of France?"), Answer(["Paris"]))
            print(d)
            ```
        """
        contexts_str = "\n\n".join([str(ctx) for ctx in self.contexts])
        reorder_contexts_str= ''
        if self.reorder_contexts is not None:
            reorder_contexts_str = "\n\n".join([str(ctx) for ctx in self.reorder_contexts])
        return f"{self.question}\n\n{self.answers}\n\nContext: \n\n{contexts_str}\nReorder contexts: \n\n{reorder_contexts_str}"
