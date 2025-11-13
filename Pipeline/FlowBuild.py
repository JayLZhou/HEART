import typing as T
from Index import get_index, get_index_config
from Storage.NameSpace import Workspace
from Chunk.DocChunk import DocChunk
from Common.ContextMixin import ContextMixin
from Option.Config2 import Config
from pydantic import BaseModel
# from Pipeline.RAGFlow import RAGFlow

class FlowBuilder(ContextMixin, BaseModel):
    """Builds different types of flows based on configuration."""
    
    def __init__(self, config: Config):
        super().__init__(config=config)
        self.workspace = Workspace(self.config.working_dir, self.config.exp_name)
        self.doc_chunk = DocChunk(self.config.chunk, self.config.token_model, self.workspace.make_for("chunk_storage"))
        self.chunk_vdb = get_index(
                get_index_config(self.config, persist_path=self.workspace.make_for("chunk_vdb").get_save_path()))
        



    def build_indexing(self, corpus):
     
        self.doc_chunk.build_chunks(corpus)
        self.chunk_vdb.build_index(self.doc_chunk.get_chunks(), [], self.config.force_rebuild)


    def build_flow(self, params: T.Dict[str, T.Any]):
        """Build the appropriate flow based on parameters.
        Only focus on online flow building.
        """
  
         
        response_synthesizer_llm = self.llm

        # get template
        template_name = params["template_name"]
        
        template = get_template(template_name)
   
        
        # build rag flow
        return self._build_rag_flow(params, response_synthesizer_llm, template)
    

    
    def _build_rag_flow(self, params: T.Dict[str, T.Any], response_synthesizer_llm, template, 
                   ):
        """Build RAG-based flow."""
       
   
     
        reranker_top_k = params.get("reranker_top_k") if params.get("reranker_enabled") else None
        
   
        
        # Build specific RAG flow type
        rag_mode = params["rag_mode"]
        common_args = {
            "retriever": rag_retriever,
            "response_synthesizer_llm": response_synthesizer_llm,
            "docstore": rag_docstore,
            "template": template,
            "reranker_top_k": reranker_top_k,
            "params": params,
        }
        
        if rag_mode == "rag":
            return RAGFlow(**common_args)
        else:
            raise ValueError(f"only 'rag' modes are supported, got: {rag_mode}")
      
