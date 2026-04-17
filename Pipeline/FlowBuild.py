import typing as T
import hashlib
import json
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
        self._dense_index_cache = {}
        self._dense_index_lock = threading.Lock()



    def build_indexing(self, corpus):
     
        self.doc_chunk.build_chunks(corpus)
        self.chunk_vdb.build_index(self.doc_chunk.get_chunks(), [], self.config.force_rebuild)
        self.sparse_index.build_index(self.doc_chunk.get_chunks(), [], self.config.force_rebuild)

    def build_flow(self, params: T.Dict[str, T.Any]):
        """Build the appropriate flow based on parameters.
        Only focus on online flow building.
        """
        # get response synthesizer llm
        # import pdb
        # pdb.set_trace()
        response_synthesizer_llm = self.get_llm(params["response_synthesizer_llm"])
        # get template
        template = get_template(params["template_name"])
        # get retriever
        retriever = self.get_retriever(
            params["rag_retriever"],
            index_params=self._extract_index_params(params),
        )
        # get reranker
                 
        reranker = get_reranker(params["reranker"])
        # build rag flow
        return RAGFlow(
            response_synthesizer_llm=response_synthesizer_llm,
            template=template,
            retriever=retriever,
            reranker=reranker,
        )
    

    def _extract_index_params(self, params: T.Dict[str, T.Any]) -> T.Dict[str, T.Any]:
        return {
            "faiss_hnsw_m": params.get("faiss_hnsw_m", self.config.faiss_hnsw_m),
            "faiss_hnsw_ef_search": params.get("faiss_hnsw_ef_search", self.config.faiss_hnsw_ef_search),
            "faiss_hnsw_ef_construction": params.get("faiss_hnsw_ef_construction", self.config.faiss_hnsw_ef_construction),
            "faiss_metric": params.get("faiss_metric", self.config.faiss_metric),
        }

    def _dense_index_signature(self, index_params: T.Dict[str, T.Any]) -> str:
        payload = json.dumps(index_params, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]

    def _get_dense_index(self, index_params: T.Dict[str, T.Any]):
        if index_params == {
            "faiss_hnsw_m": self.config.faiss_hnsw_m,
            "faiss_hnsw_ef_search": self.config.faiss_hnsw_ef_search,
            "faiss_hnsw_ef_construction": self.config.faiss_hnsw_ef_construction,
            "faiss_metric": self.config.faiss_metric,
        }:
            return self.chunk_vdb

        signature = self._dense_index_signature(index_params)
        with self._dense_index_lock:
            if signature in self._dense_index_cache:
                return self._dense_index_cache[signature]

            persist_path = self.workspace.make_for(f"chunk_vdb_{signature}").get_save_path()
            dense_config = self.config.model_copy(
                update={
                    "faiss_hnsw_m": index_params["faiss_hnsw_m"],
                    "faiss_hnsw_ef_search": index_params["faiss_hnsw_ef_search"],
                    "faiss_hnsw_ef_construction": index_params["faiss_hnsw_ef_construction"],
                    "faiss_metric": index_params["faiss_metric"],
                }
            )
            index = get_index(get_index_config(dense_config, persist_path=persist_path))
            index.build_index(self.doc_chunk.get_chunks(), [], self.config.force_rebuild)
            self._dense_index_cache[signature] = index
            return index

    def get_retriever(self, params, index_params: T.Dict[str, T.Any] | None = None):
        index_params = index_params or self._extract_index_params({})
        retrievers = []
        if params["method"] == "dense" or params["method"] == "hybrid":
            dense_index = self._get_dense_index(index_params)
            retrievers.append(dense_index.get_retriever(params["top_k"]))
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

      
