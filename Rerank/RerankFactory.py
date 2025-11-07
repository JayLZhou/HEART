import typing as T
import time
from typing import Optional, Dict, Any, List
from llama_index.core.query_engine import QueryBundle
from llama_index.core.schema import NodeWithScore
try:
    from Schema.DocumentSchema import Document, Question, Answer, Context
except ImportError:
    # Fallback for syftr project structure
    try:
        from syftr.Schema.DocumentSchema import Document, Question, Answer, Context
    except ImportError:
        raise ImportError("Cannot import Document, Question, Answer, Context. Please check your project structure.")
from Common.Logger import logger

from Rerank.Upr import UPR
from Rerank.ColbertRanker import ColBERTReranker
from Rerank.Flashrank import FlashRanker
from Rerank.Monot5 import MonoT5
from Rerank.Rankt5 import RankT5
from Rerank.Echorank import EchoRankReranker
from Rerank.Listt5 import ListT5
from Rerank.Twolar import TWOLAR
from Rerank.TransformerRanker import TransformerRanker
from Rerank.Monobert import MonoBERT
from Rerank.InRanker import InRanker
from Rerank.Reranking import Reranking

METHOD_MAP = {
    # Existing reranking methods
    'upr': UPR,
    'flashrank': FlashRanker,
    'monot5': MonoT5,
    'rankt5': RankT5,
    'listt5': ListT5,
    'transformer_ranker': TransformerRanker,
    'colbert_ranker': ColBERTReranker,
    'twolar': TWOLAR,
    'echorank': EchoRankReranker,
    'monobert_ranker': MonoBERT,
    "inranker": InRanker
}




def get_reranker(method: str, model_name: str, api_key: str = None, **kwargs):
    """Factory method to create a reranker instance"""
    if method not in METHOD_MAP:
        raise ValueError(f"Unknown reranker method: {method}. Available: {list(METHOD_MAP.keys())}")
    return METHOD_MAP[method](method=method, model_name=model_name, api_key=api_key, **kwargs)


class BatchReranker:
    """通用批处理Reranker - 兼容预定义模型和多种实现"""

    def __init__(self, reranker_name: str, top_k: int, base_params: Optional[Dict[str, Any]] = None):
        self.reranker_name = reranker_name
        self.top_k = top_k

        logger.info(f"🚀 初始化批处理Reranker: {reranker_name}, top_k={top_k}")

        # 合并基础参数
        params: Dict[str, Any] = {}
        if base_params:
            params.update(base_params)

        params.update({
            "reranker_enabled": True,
            "reranker_llm_name": reranker_name,
            "reranker_llm": reranker_name,
            "reranker_top_k": top_k,
        })

        # 使用工厂创建后处理器
        postprocessor = build_reranker_postprocessor(params)

        if postprocessor is None:
            raise ValueError(f"无法创建reranker: {reranker_name}")

        self._postprocessor: BaseNodePostprocessor = postprocessor
        logger.info("✅ 批处理reranker初始化成功")

    def batch_rerank_all_queries(
        self, 
        all_retrieved_nodes: List[List[NodeWithScore]],
        queries: List[str]
    ) -> List[List[NodeWithScore]]:
        """批量执行reranking"""
        start_time = time.time()
        logger.info(f"🚀 BatchReranker开始处理: {len(queries)}个查询")
        logger.info(f"📊 Reranker配置: 模型={self.reranker_name}, top_k={self.top_k}")

        results: List[List[NodeWithScore]] = []
        for idx, (query, nodes) in enumerate(zip(queries, all_retrieved_nodes)):
            if not nodes:
                results.append([])
                continue

            try:
                query_bundle = QueryBundle(query)
                reranked_nodes = self._postprocessor.postprocess_nodes(nodes, query_bundle=query_bundle)
                results.append(reranked_nodes[:self.top_k] if reranked_nodes else [])
                logger.debug(f"✅ Reranker '{self.reranker_name}' 成功处理查询索引 {idx}")
            except Exception as exc:
                logger.warning(f"⚠️ Rerank失败(索引 {idx}, 模型 {self.reranker_name}): {exc}")
                results.append(nodes[:self.top_k])

        total_time = time.time() - start_time
        output_nodes = sum(len(nodes) for nodes in results)
        successful_queries = len([nodes for nodes in results if nodes])

        logger.info(f"✅ BatchReranker处理完成")
        logger.info(f"📊 输出统计: 成功查询数={successful_queries}/{len(queries)}, 输出节点数={output_nodes}")
        if total_time > 0:
            logger.info(f"📊 处理效率: 总耗时={total_time:.2f}s, 处理速度={len(queries)/total_time:.1f} queries/s")

        return results


class UserRerankAdapter(BaseNodePostprocessor):
    """Adapter to integrate user's reranker models with LlamaIndex's postprocessor interface"""

    def __init__(self, reranker_name: str, model_name: str = None, top_n: int = 5, **kwargs):
        super().__init__()
        self._reranker_name = reranker_name.lower()
        self._model_name = model_name
        self._top_n = top_n
        self._kwargs = kwargs
        self._reranker = None
        self._initialize_reranker()

    def _initialize_reranker(self):
        """延迟初始化用户的reranker实例"""
        try:
            if not USER_RERANK_AVAILABLE:
                raise ValueError("User's Rerank framework not available. Please check dependencies.")

            if self._reranker_name not in METHOD_MAP:
                raise ValueError(f"Unknown reranker: {self._reranker_name}. Available: {list(METHOD_MAP.keys())}")

            init_params = {'method': self._reranker_name}
            if self._model_name and self._model_name != "default":
                init_params['model_name'] = self._model_name

            # 只添加支持的参数
            supported_params = ['device', 'batch_size', 'api_key']
            for param_name in supported_params:
                if param_name in self._kwargs:
                    init_params[param_name] = self._kwargs[param_name]

            self._reranker = Reranking(**init_params)
            logger.info(f"✅ Successfully initialized {self._reranker_name} reranker with model {self._model_name}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize {self._reranker_name} reranker: {e}")
            self._reranker = None
            raise

    def _convert_to_user_format(self, nodes: T.List[NodeWithScore], query: str) -> Document:
        """Convert LlamaIndex nodes to user's Document format"""
        question = Question(query)
        answers = Answer([])
        contexts = []

        for i, node in enumerate(nodes):
            context = Context(
                text=node.get_content(),
                id=i,
                score=getattr(node, 'score', 0.0)
            )
            contexts.append(context)

        return Document(question=question, answers=answers, contexts=contexts)

    def _convert_from_user_format(
        self, 
        document: Document, 
        original_nodes: T.List[NodeWithScore]
    ) -> T.List[NodeWithScore]:
        """Convert user's reranked results back to LlamaIndex format"""
        content_to_node = {node.get_content(): node for node in original_nodes}

        reranked_nodes = []
        for context in document.reorder_contexts[:self._top_n]:
            original_node = content_to_node.get(context.text)
            if original_node:
                if hasattr(context, 'score'):
                    original_node.score = context.score
                reranked_nodes.append(original_node)

        return reranked_nodes

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


def build_reranker_postprocessor(params: T.Dict[str, T.Any]) -> T.Optional[BaseNodePostprocessor]:
    """Factory method to build reranker postprocessor based on configuration"""
    if not params.get("reranker_enabled"):
        return None

    reranker_name = params.get("reranker_llm_name") or params.get("reranker_model", "flashrank")
    reranker_top_k = params.get("reranker_top_k", 5)

    logger.info(f"🔧 Building reranker: {reranker_name}, top_k={reranker_top_k}")



    try:
        model_name = params.get("reranker_model_name")
        if not model_name or model_name == "default":
            model_defaults = {
                'flashrank': 'ms-marco-MiniLM-L-12-v2',
                'transformer_ranker': 'mxbai-rerank-xsmall',
                'colbert_ranker': 'Colbert',
                'monot5': 'monot5-base-msmarco',
                'rankt5': 'rankt5-base',
                'listt5': 'listt5-base',
                'twolar': 'twolar-xl',
                'monobert_ranker': 'monobert-large',
                'inranker': 'inranker-base',
                'echorank': 'flan-t5-large',
                'upr': 't5-base'
            }
            model_name = model_defaults.get(user_reranker_name)

        # 智能GPU分配
        try:
            from BOTuner.OptunaTuner import DEVICE_ID
            optimal_device = f"cuda:{DEVICE_ID}"
            device = params.get("reranker_device", optimal_device)
        except (ImportError, AttributeError):
            device = params.get("reranker_device", "cuda")

        return UserRerankAdapter(
            reranker_name=user_reranker_name,
            model_name=model_name or "default",
            top_n=reranker_top_k,
            device=device,
            batch_size=params.get("reranker_batch_size", 16)
        )
    except Exception as e:
        logger.error(f"❌ Failed to create {user_reranker_name}: {e}")

    return None
