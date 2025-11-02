from Index import get_index, get_index_config
from Storage.NameSpace import Workspace
from Chunk.DocChunk import DocChunk

class FlowBuilder:
    """Builds different types of flows based on configuration."""
    
    def __init__(self, study_config: StudyConfig):
        self.study_config = study_config
        self.workspace = Workspace(self.config.working_dir, self.config.index_name)
        self.doc_chunk = DocChunk(self.config.chunk, self.config.token_model, self.workspace.make_for("chunk_storage"))
        self.chunk_vdb = get_index(
                get_index_config(self.config, persist_path=self.chunk_vdb_namespace.get_save_path()))
        
    

    def build_indexing(self):
        self.doc_chunk.build_chunks(docs)
        self.chunk_vdb.insert()
        pass

    def build_flow(self, params: T.Dict[str, T.Any]) -> Flow:
        """Build the appropriate flow based on parameters."""
        from hammer.llm import get_llm
        # import pdb
        # pdb.set_trace()
        
        # 🔧 修复LLM名称中的引号问题
        def clean_llm_name(name):
            """清理LLM名称中的多余引号"""
            if isinstance(name, str):
                # 去掉前后的双引号和单引号
                return name.strip().strip('"').strip("'")
            return name
        
        response_synthesizer_llm = get_llm(clean_llm_name(params["response_synthesizer_llm"]))
        enforce_full_evaluation = params.get("enforce_full_evaluation", False)
        
        

        
        # Build appropriate flow type
        template_name = params["template_name"]
        is_few_shot = get_qa_examples is not None
      
        template = get_template(template_name, with_context=do_rag, with_few_shot_prompt=is_few_shot)
   
        
    
        return self._build_rag_flow(params, response_synthesizer_llm, template, 
                                      get_qa_examples, enforce_full_evaluation)
    

    
    def _build_rag_flow(self, params: T.Dict[str, T.Any], response_synthesizer_llm, template, 
                       get_qa_examples, enforce_full_evaluation: bool) -> Flow:
        """Build RAG-based flow."""
        from hammer.llm import get_llm
        
        # 🔧 修复查询转换检测问题：需要创建LLM实例来生成TransformQueryEngine包装器
        # 虽然实际的LLM调用由BatchLLMCaller处理，但包装器检测逻辑需要这些实例存在
        hyde_llm = None
        if params.get("hyde_enabled", False):
            hyde_llm_name = params.get("hyde_llm_name", "Qwen2_5-7b")
            try:
                # 清理LLM名称中的引号
                clean_name = hyde_llm_name.strip().strip('"').strip("'")
                hyde_llm = get_llm(clean_name)
                logger.debug(f"✅ 创建HyDE LLM实例用于TransformQueryEngine包装: {clean_name}")
            except Exception as e:
                logger.warning(f"⚠️ 创建HyDE LLM实例失败，将跳过HyDE功能: {e}")
                hyde_llm = None
        
        # 🎯 专用reranker不需要LLM实例，跳过创建
        reranker_llm = None
        if params.get("reranker_enabled", False):
            logger.info("🎯 检测到专用reranker，跳过LLM实例创建")
        
        reranker_top_k = params.get("reranker_top_k") if params.get("reranker_enabled") else None
        additional_context_num_nodes = params.get("additional_context_num_nodes", 0) if params.get("additional_context_enabled") else 0
        
        rag_retriever, rag_docstore = build_rag_retriever(self.study_config, params)
        
        # Build specific RAG flow type
        rag_mode = params["rag_mode"]
        common_args = {
            "retriever": rag_retriever,
            "response_synthesizer_llm": response_synthesizer_llm,
            "docstore": rag_docstore,
            "template": template,
            "get_examples": get_qa_examples,
            "hyde_llm": hyde_llm,  # 🔧 修复：现在根据配置创建真实的LLM实例
            "reranker_llm": reranker_llm,  # 🔧 修复：现在根据配置创建真实的LLM实例
            "reranker_top_k": reranker_top_k,
            "additional_context_num_nodes": additional_context_num_nodes,
            "enforce_full_evaluation": enforce_full_evaluation,
            "params": params,
        }
        
        if rag_mode == "rag":
            return RAGFlow(**common_args)
        else:
            raise ValueError(f"only 'rag' modes are supported, got: {rag_mode}")
      
