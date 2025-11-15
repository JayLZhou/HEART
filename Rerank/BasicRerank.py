
from abc import abstractmethod
from Schema.DocumentSchema import Document
from llama_index.core.postprocessor.types import BaseNodePostprocessor
import typing as T
from llama_index.core.schema import NodeWithScore, QueryBundle
from Common.Logger import logger
import time
from typing import List

class BaseRanking(BaseNodePostprocessor):
    """
    An abstract base class for implementing different ranking models.

    This class defines the interface for all ranking models, ensuring that all subclasses implement the required methods.

    Attributes:
        method (str): The name of the ranking method.
        model_name (str): The name of the model being used for ranking.
        api_key (str, optional): An optional API key for accessing remote models or services.
    """

    @abstractmethod
    def __init__(self, method: str= None, model_name: str= None, api_key: str= None, **kwargs) ->None:
        """
        Initializes the base ranking model.

        Args:
            method (str, optional): The name of the ranking method. Defaults to None.
            model_name (str, optional): The name of the model being used for ranking. Defaults to None.
            api_key (str, optional): An optional API key for accessing remote models or services. Defaults to None.

        Example:
            ```python
            class MyRanking(BaseRanking):
                def __init__(self, method, model_name):
                    super().__init__(method, model_name)
            ```
        """
        pass

    @abstractmethod
    def rank(self, documents: list[Document]):
        """
        Abstract method to rank a list of documents.

        Args:
            documents (list[Document]): A list of Document instances that need to be ranked.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.

        Example:
            ```python
            class MyRanking(BaseRanking):
                def __init__(self, method, model_name):
                    super().__init__(method, model_name)

                def rank(self, documents):
                    # Ranking implementation here
                    pass
            ```
        """
        pass
    
    # def _convert_to_user_format(self, nodes: T.List[NodeWithScore], query: str) -> Document:
    #     """Convert LlamaIndex nodes to user's Document format"""
    #     question = Question(query)
    #     answers = Answer([])
    #     contexts = []

    #     for i, node in enumerate(nodes):
    #         context = Context(
    #             text=node.get_content(),
    #             id=i,
    #             score=getattr(node, 'score', 0.0)
    #         )
    #         contexts.append(context)

    #     return Document(question=question, answers=answers, contexts=contexts)

    # def _convert_from_user_format(
    #     self, 
    #     document: Document, 
    #     original_nodes: T.List[NodeWithScore]
    # ) -> T.List[NodeWithScore]:
    #     """Convert user's reranked results back to LlamaIndex format"""
    #     content_to_node = {node.get_content(): node for node in original_nodes}

    #     reranked_nodes = []
    #     for context in document.reorder_contexts[:self._top_n]:
    #         original_node = content_to_node.get(context.text)
    #         if original_node:
    #             if hasattr(context, 'score'):
    #                 original_node.score = context.score
    #             reranked_nodes.append(original_node)

    #     return reranked_nodes

    def _postprocess_nodes(
        self,
        nodes: T.List[NodeWithScore],
        query_bundle: T.Optional[QueryBundle] = None,
    ) -> T.List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Query bundle is required for reranking.")
        if not nodes:
            return []

        if self._reranker is None:
            logger.error(f"❌ Reranker not initialized for {self._reranker_name}")
            return nodes[:self._top_n]

        try:
            document = self._convert_to_user_format(nodes, query_bundle.query_str)
            reranked_documents = self._reranker.rank([document])
            return self._convert_from_user_format(reranked_documents[0], nodes)
        except Exception as e:
            logger.error(f"❌ Reranking failed with {self._reranker_name}: {e}")
            return nodes[:self._top_n]

    def batch_rerank_all_queries(
        self, 
        all_retrieved_nodes: List[List[NodeWithScore]],
        queries: List[str]
    ) -> List[List[NodeWithScore]]:
        """批量执行reranking"""
        start_time = time.time()
        logger.info(f"🚀 BatchReranker开始处理: {len(queries)}个查询")
        logger.info(f"📊 Reranker配置: 模型={self.method}, top_k={self.top_k}")

        results: List[List[NodeWithScore]] = []
        for idx, (query, nodes) in enumerate(zip(queries, all_retrieved_nodes)):
            if not nodes:
                results.append([])
                continue

            try:
                query_bundle = QueryBundle(query)
                reranked_nodes = self._postprocessor.postprocess_nodes(nodes, query_bundle=query_bundle)
                results.append(reranked_nodes[:self.top_k] if reranked_nodes else [])
                logger.debug(f"✅ Reranker '{self.method}' 成功处理查询索引 {idx}")
            except Exception as exc:
                logger.warning(f"⚠️ Rerank失败(索引 {idx}, 模型 {self.method}): {exc}")
                results.append(nodes[:self.top_k])

        total_time = time.time() - start_time
        output_nodes = sum(len(nodes) for nodes in results)
        successful_queries = len([nodes for nodes in results if nodes])

        logger.info(f"✅ BatchReranker处理完成")
        logger.info(f"📊 输出统计: 成功查询数={successful_queries}/{len(queries)}, 输出节点数={output_nodes}")
        if total_time > 0:
            logger.info(f"📊 处理效率: 总耗时={total_time:.2f}s, 处理速度={len(queries)/total_time:.1f} queries/s")

        return results
        