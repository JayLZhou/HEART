import typing as T
from Index import get_index, get_index_config
from Storage.NameSpace import Workspace
from Chunk.DocChunk import DocChunk
from Common.ContextMixin import ContextMixin
from Option.Config2 import Config
from pydantic import BaseModel
from Prompt import get_template
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from Rerank import get_reranker
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



    def build_indexing(self, corpus):
     
        self.doc_chunk.build_chunks(corpus)
        self.chunk_vdb.build_index(self.doc_chunk.get_chunks(), [], self.config.force_rebuild)
        self.sparse_index.build_index(self.doc_chunk.get_chunks(), [], self.config.force_rebuild)

    def build_flow(self, params: T.Dict[str, T.Any], config: Config):
        """Build the appropriate flow based on parameters.
        Only focus on online flow building.
        """
  

        # get response synthesizer llm
        response_synthesizer_llm = self.get_llm(params["response_synthesizer_llm"])
        # get template
        template = get_template(params["template_name"])
        # build rag flow
        import pdb
        retrievers = self.get_retriever(params["rag_retriever"])
        # get reranker
        pdb.set_trace()
        reranker = get_reranker(params["reranker"])
    
        self._flow =  RAGFlow(**common_args)
    

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

      
