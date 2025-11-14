import typing as T
from Index import get_index, get_index_config
from Storage.NameSpace import Workspace
from Chunk.DocChunk import DocChunk
from Common.ContextMixin import ContextMixin
from Option.Config2 import Config
from pydantic import BaseModel
from Prompt import get_template
# from Pipeline.RAGFlow import RAGFlow

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
        pdb.set_trace()
        retriever = get_retriever(params["rag_retriever"])
        self._build_rag_flow(params, response_synthesizer_llm, template)
    

    
    def _build_rag_flow(self, params: T.Dict[str, T.Any], response_synthesizer_llm, template):
        """Build RAG-based flow."""
       
   
     
        reranker_top_k = params.get("reranker_top_k") if params.get("reranker_enabled") else None
        import pdb
        pdb.set_trace()
   
        
        # Build specific RAG flow type
  
        common_args = {
            "retriever": rag_retriever,
            "response_synthesizer_llm": response_synthesizer_llm,
            "template": template,
            "reranker_top_k": reranker_top_k,
            "params": params,
        }
        
        import pdb
        pdb.set_trace()
    
        self.flow =  RAGFlow(**common_args)

      
