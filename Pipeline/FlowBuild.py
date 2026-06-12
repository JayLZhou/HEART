import typing as T
import hashlib
import threading
from Index import get_index, get_index_config
from Storage.NameSpace import Workspace
from Chunk.DocChunk import DocChunk
from Common.ContextMixin import ContextMixin
from Option.Config2 import Config
from pydantic import BaseModel
from Prompt import get_template
from Index.FusionRetriever import QueryFusionRetriever, FUSION_MODES
from Rerank import get_reranker
from Pipeline.RAGFlow import RAGFlow
class FlowBuilder(ContextMixin, BaseModel):
    """Builds different types of flows based on configuration."""
    
    def __init__(self, config: Config):
        super().__init__(config=config)
        self.workspace = Workspace(self.config.working_dir, self.config.exp_name)
        self.doc_chunk = DocChunk(self.config.chunk, self.config.token_model, self.workspace.make_for("chunk_storage"))
        self.chunk_vdb = get_index(
                get_index_config(self.config, persist_path=self.workspace.make_for("chunk_vdb").get_save_path()))
        self.sparse_index = get_index(
                get_index_config(self.config, persist_path=self.workspace.make_for("sparse_index").get_save_path(), type="sparse"))
        self._dense_chunks = None
        self._dense_index_signature = None
        self._dense_index_lock = threading.Lock()



    def build_indexing(self, corpus):
      
        self.doc_chunk.build_chunks(corpus)
        self._dense_chunks = self.doc_chunk.get_chunks()
        self._ensure_dense_index({})
        self.sparse_index.build_index(self.doc_chunk.get_chunks(), [], self.config.force_rebuild)

    def build_flow(self, params: T.Dict[str, T.Any]):
        """Build the appropriate flow based on parameters.
        Only focus on online flow building.
        """
        self._ensure_dense_index(params)
        # get response synthesizer llm
        # import pdb
        # pdb.set_trace()
        response_synthesizer_llm = self.get_llm(params["response_synthesizer_llm"])
        # get template
        template = get_template(params["template_name"])
        # get retriever
        retriever = self.get_retriever(params["rag_retriever"])
        # get reranker
                 
        reranker = get_reranker(params["reranker"])
        # build rag flow
        return RAGFlow(
            response_synthesizer_llm=response_synthesizer_llm,
            template=template,
            retriever=retriever,
            reranker=reranker,
        )
    

    def get_retriever(self, params):
        retrievers = []
        if params["method"] == "dense" or params["method"] == "hybrid":
            retrievers.append(self.chunk_vdb.get_retriever(params["top_k"]))
        if params["method"] == "sparse" or params["method"] == "hybrid":
            retrievers.append(self.sparse_index.get_retriever(params["top_k"]))
        if params["method"] == "hybrid":
            hybrid_bm25_weight = float(params["hybrid_bm25_weight"])
            retriever_weights = [hybrid_bm25_weight, 1 - hybrid_bm25_weight]
        else:
            return retrievers[0]

        fusion_retriever_params = {
            "llm": self.llm,
            "mode": FUSION_MODES(params["fusion_mode"]),
            "use_async": False,
            "verbose": True,
            "similarity_top_k": params["top_k"],
            "num_queries": 1,
            "retriever_weights": retriever_weights,
            "retrievers": retrievers,
        }
        if params["query_decomposition_enabled"] == True:
            query_decomposition_llm = self.get_llm(params["query_decomposition_llm_name"])
            fusion_retriever_params.update(
                **{
                    "llm": query_decomposition_llm,
                    "num_queries": params["query_decomposition_num_queries"],
                }
            )
   
        return  QueryFusionRetriever(**fusion_retriever_params)

    def _effective_faiss_params(self, params: T.Dict[str, T.Any]) -> dict[str, T.Any]:
        return {
            "faiss_hnsw_m": int(params.get("faiss_hnsw_m", self.config.faiss_hnsw_m)),
            "faiss_hnsw_ef_search": int(params.get("faiss_hnsw_ef_search", self.config.faiss_hnsw_ef_search)),
            "faiss_hnsw_ef_construction": int(
                params.get("faiss_hnsw_ef_construction", self.config.faiss_hnsw_ef_construction)
            ),
            "faiss_metric": str(params.get("faiss_metric", self.config.faiss_metric)),
        }

    def _dense_index_persist_path(self, faiss_params: dict[str, T.Any]) -> str:
        signature = hashlib.md5(
            repr(sorted(faiss_params.items())).encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()[:12]
        return self.workspace.make_for("chunk_vdb").get_save_path(resource_name=signature)

    def _ensure_dense_index(self, params: T.Dict[str, T.Any]) -> None:
        with self._dense_index_lock:
            if self.config.vdb_type != "faiss":
                if self._dense_chunks is not None and self._dense_index_signature is None:
                    self.chunk_vdb.build_index(self._dense_chunks, [], self.config.force_rebuild)
                    self._dense_index_signature = "non_faiss"
                return

            if self._dense_chunks is None:
                return

            faiss_params = self._effective_faiss_params(params)
            signature = repr(sorted(faiss_params.items()))
            if signature == self._dense_index_signature:
                return

            index_config = self.config.model_copy(
                update=faiss_params,
                deep=True,
            )
            persist_path = self._dense_index_persist_path(faiss_params)
            self.chunk_vdb = get_index(get_index_config(index_config, persist_path=persist_path))
            self.chunk_vdb.build_index(self._dense_chunks, [], self.config.force_rebuild)
            self._dense_index_signature = signature

      
