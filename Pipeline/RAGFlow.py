import time
import typing as T
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import cached_property
import llama_index.core.instrumentation as instrument


from llama_index.core import (
    PromptTemplate,
    QueryBundle,
    Response,
    get_response_synthesizer,
)

from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.indices.query.query_transform.base import (
    HyDEQueryTransform,
)
from llama_index.core.llms import ChatMessage, CompletionResponse, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor import LLMRerank, PrevNextNodePostprocessor
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import PromptType
from llama_index.core.query_engine import (
    BaseQueryEngine,
    RetrieverQueryEngine,
    SubQuestionQueryEngine,
    TransformQueryEngine,
)
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.tools import BaseTool, QueryEngineTool, ToolMetadata
from numpy import ceil

from hammer.configuration import cfg
from hammer.instrumentation.arize import instrument_arize
from hammer.llm import get_llm_name, get_tokenizer
from hammer.logger import logger
# from hammer.studies import get_critique_template, get_react_template
from hammer.rerankers.enhanced_factory import build_reranker_postprocessor

class RAGFlow():
    response_synthesizer_llm: LLM | FunctionCallingLLM
    template: str | None = None
    name: str = "RAG Flow"
    params: dict | None = None


    def __repr__(self):
        return f"{self.name}: {self.params}"

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
        if self.get_examples is not None:
            function_mappings = {"few_shot_examples": self.get_examples}

        logger.debug("Creating prompt template from '%s'", self.template)
        prompt_template = PromptTemplate(
            template=self.template,
            prompt_type=PromptType.QUESTION_ANSWER,
            function_mappings=function_mappings,
        )

        return prompt_template

    @cached_property
    def query_engine(self) -> BaseQueryEngine:
        node_postprocessors: T.List[BaseNodePostprocessor] = []
        if self.params and self.params.get("reranker_enabled"):
            reranker = build_reranker_postprocessor(self.params)
            if reranker:
                node_postprocessors.append(reranker)
       
        response_synthesizer = get_response_synthesizer(
            llm=self.response_synthesizer_llm,
            response_mode=ResponseMode.COMPACT,
            text_qa_template=self.prompt_template,
        )
        retriever = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors,
        )
      
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
        return self.query_engine.retrieve(QueryBundle(query))

    def aretrieve(self, query: str) -> T.List[NodeWithScore]:
        assert hasattr(self.query_engine, "aretrieve"), (
            f"{self.query_engine} does not have 'aretrieve' method"
        )

        return self.query_engine.aretrieve(QueryBundle(query))

    def _generate(
        self, query: str, invocation_id: str
    ) -> T.Tuple[CompletionResponse, float]:
        start_time = time.perf_counter()
        response = self.query_engine.query(query)
        assert isinstance(response, Response), (
            f"Expected Response, got {type(response)=}"
        )
        completion_response = CompletionResponse(
            text=str(response.response),
            additional_kwargs={
                "source_nodes": response.source_nodes,
                **(response.metadata or {}),  # type: ignore
            },
        )
        duration = time.perf_counter() - start_time
        return completion_response, duration

    def _agenerate(
        self, query: str, invocation_id: str
    ) -> T.Tuple[CompletionResponse, float]:
        start_time = time.perf_counter()
        response = self.query_engine.aquery(query)
        assert isinstance(response, Response), (
            f"Expected Response, got {type(response)=}"
        )
        assert isinstance(response.response, str), (
            f"Expected str, got {type(response.response)=}"
        )
        completion_response = CompletionResponse(
            text=str(response.response),
            additional_kwargs={
                "source_nodes": response.source_nodes,
                **(response.metadata or {}),
            },
        )
        duration = time.perf_counter() - start_time
        return completion_response, duration
