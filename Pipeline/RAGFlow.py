from functools import cached_property
from llama_index.core import (
    PromptTemplate,
    QueryBundle,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptType
from llama_index.core.query_engine import (
    BaseQueryEngine,
    RetrieverQueryEngine,
)
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from Common.Logger import logger
from Rerank.Reranking import Reranker
import typing as T
from Common.Utils import extract_answer
class RAGFlow:
    def __init__(self, response_synthesizer_llm: LLM | FunctionCallingLLM, template: str, retriever: BaseRetriever, reranker: Reranker):
        self.response_synthesizer_llm = response_synthesizer_llm
        self.template = template
        self.retriever = retriever
        self.reranker = reranker



    @property
    def verbose(self):
        log_level = logger.level
        if log_level <= 20:
            return True
        return False

    @property
    def prompt_template(self) -> PromptTemplate:
        if self.template is None:
            raise ValueError("Flow template not set. Cannot create prompt template.")
        prompt_template = None
        function_mappings = None
    
        logger.debug("Creating prompt template from '%s'", self.template)

        prompt_template = PromptTemplate(
            template=self.template,
            prompt_type=PromptType.QUESTION_ANSWER,
            function_mappings=function_mappings,
        )

        return prompt_template

    @cached_property
    def query_engine(self) -> BaseQueryEngine:

        retriever = RetrieverQueryEngine(
            retriever=self.retriever)
      
        return retriever

    def get_prompt(self, query) -> str:
        if self.template is None:
            return query

        if self.get_examples is None:
            return self.template.format(query_str=query)

        examples = self.get_examples(query)
        assert examples, "No examples found for few-shot prompting"

        return self.template.format(
            query_str=query,
            few_shot_examples=examples,
        )

    def retrieve(self, query: str) -> T.List[NodeWithScore]:
        return self.retriever.retrieve(QueryBundle(query))  

    def query(self, query: str) -> str:
        # Generate response
        retrieved_nodes = self.retrieve(query)
        reranked_nodes = self.reranker.rerank(retrieved_nodes, query)
        context = "\n".join([node.text for node in reranked_nodes])
        instruction = self.prompt_template.format(query=query, context=context)
        response = self.response_synthesizer_llm.ask(msg=instruction)
        response = extract_answer(response)
        return response


