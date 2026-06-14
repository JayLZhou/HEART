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
from Prompt import get_synthesis_prompts


class RAGFlow:
    def __init__(
        self,
        response_synthesizer_llm: LLM | FunctionCallingLLM,
        template: str,
        retriever: BaseRetriever,
        reranker: Reranker,
        synthesis_mode: str = "direct",
        intermediate_length: int = 100,
    ):
        self.response_synthesizer_llm = response_synthesizer_llm
        self.template = template
        self.retriever = retriever
        self.reranker = reranker
        self.synthesis_mode = synthesis_mode
        self.intermediate_length = intermediate_length
        self._synthesis_prompts = get_synthesis_prompts()

    @property
    def verbose(self):
        return logger.level <= 20

    @property
    def prompt_template(self) -> PromptTemplate:
        if self.template is None:
            raise ValueError("Flow template not set. Cannot create prompt template.")
        return PromptTemplate(
            template=self.template,
            prompt_type=PromptType.QUESTION_ANSWER,
        )

    @cached_property
    def query_engine(self) -> BaseQueryEngine:
        return RetrieverQueryEngine(retriever=self.retriever)

    def retrieve(self, query: str) -> T.List[NodeWithScore]:
        return self.retriever.retrieve(QueryBundle(query))

    def _synthesize_direct(self, query: str, nodes: T.List[NodeWithScore]) -> str:
        context = "\n".join([node.text for node in nodes])
        instruction = self.prompt_template.format(query=query, context=context)
        response = self.response_synthesizer_llm.ask(msg=instruction)
        return extract_answer(response)

    def _synthesize_map_reduce(self, query: str, nodes: T.List[NodeWithScore]) -> str:
        map_tmpl = self._synthesis_prompts['map']
        reduce_tmpl = self._synthesis_prompts['reduce']

        partial_answers = []
        for node in nodes:
            try:
                instruction = map_tmpl.format(query=query, context=node.text)
                answer = self.response_synthesizer_llm.ask(
                    msg=instruction, max_tokens=self.intermediate_length
                )
                partial_answers.append(answer.strip())
            except Exception as e:
                logger.warning(f"Map step failed for a chunk: {e}")

        if not partial_answers:
            return ""

        combined = "\n\n".join(
            f"Partial answer {i+1}: {a}" for i, a in enumerate(partial_answers)
        )
        reduce_instruction = reduce_tmpl.format(query=query, context=combined)
        response = self.response_synthesizer_llm.ask(msg=reduce_instruction)
        return extract_answer(response)

    def _synthesize_refine(self, query: str, nodes: T.List[NodeWithScore]) -> str:
        if not nodes:
            return ""

        refine_tmpl = self._synthesis_prompts['refine']

        # Bootstrap with first chunk using the main template
        context = nodes[0].text
        instruction = self.prompt_template.format(query=query, context=context)
        current_answer = self.response_synthesizer_llm.ask(
            msg=instruction, max_tokens=self.intermediate_length
        )

        # Iteratively refine with remaining chunks
        for node in nodes[1:]:
            try:
                instruction = refine_tmpl.format(
                    query=query,
                    existing_answer=current_answer.strip(),
                    context=node.text,
                )
                current_answer = self.response_synthesizer_llm.ask(
                    msg=instruction, max_tokens=self.intermediate_length
                )
            except Exception as e:
                logger.warning(f"Refine step failed for a chunk: {e}")

        return extract_answer(current_answer)

    def query(self, query: str) -> str:
        retrieved_nodes = self.retrieve(query)
        reranked_nodes = self.reranker.rerank(retrieved_nodes, query)

        mode = (self.synthesis_mode or "direct").lower()
        if mode == "map_reduce":
            return self._synthesize_map_reduce(query, reranked_nodes)
        elif mode == "refine":
            return self._synthesize_refine(query, reranked_nodes)
        else:
            return self._synthesize_direct(query, reranked_nodes)
