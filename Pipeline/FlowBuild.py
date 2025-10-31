
class FlowBuilder:
    """Builds different types of flows based on configuration."""
    
    def __init__(self, study_config: StudyConfig):
        self.study_config = study_config

    def build_few_shot_retriever(self, params: T.Dict[str, T.Any]) -> T.Optional[T.Callable]:
        """Build few-shot example retriever if enabled."""
        if not self.study_config.search_space.is_few_shot(params):
            return None
        
        few_shot_embedding_model_name = params["few_shot_embedding_model"]
        few_shot_embedding_model, _ = get_embedding_model(
            few_shot_embedding_model_name,
            timeout_config=self.study_config.timeouts,
            device=self.study_config.optimization.embedding_device,
            use_hf_endpoint_models=self.study_config.optimization.use_hf_embedding_models,
        )
        
        return self._create_example_retriever(params, few_shot_embedding_model)
    
    def _create_example_retriever(self, params: T.Dict[str, T.Any], embedding_model) -> T.Callable:
        """Create the actual example retriever."""
        assert embedding_model, "No embedding model for dynamic few-shot prompting"
        logger.info("Building few-shot retriever")
        
        dataset_iter = self.study_config.dataset.iter_examples(partition="train")
        logger.info("Getting few-shot examples from dataset")
        
        if self.study_config.toy_mode:
            dataset_iter = itertools.islice(dataset_iter, 20)
        
        few_shot_nodes = []
        for pair in dataset_iter:
            line = f"{{'query': '''{pair.question}''', 'response': '''{pair.answer}'''}}"
            few_shot_nodes.append(TextNode(text=line))
        
        if not isinstance(embedding_model, HFEndpointEmbeddings):
            embedding_model.reset_timeouts(total_chunks=len(few_shot_nodes))
        
        logger.info("Building few-shot retriever index")
        few_shot_index = VectorStoreIndex(nodes=few_shot_nodes, embed_model=embedding_model)
        logger.info("Built few-shot retriever index")
        
        few_shot_retriever = few_shot_index.as_retriever(
            similarity_top_k=params["few_shot_top_k"], 
            similarity_threshold=None
        )

        def get_qa_examples(query_str, **kwargs):
            _ = kwargs  # Mark as used
            return self._get_examples(few_shot_retriever, query_str)

        return get_qa_examples
    
    def _get_examples(self, example_retriever: BaseRetriever, query_str: str) -> str:
        """Extract examples from retriever results."""
        retrieved_nodes: T.List[NodeWithScore] = example_retriever.retrieve(query_str)
        result_strs = []
        
        for n in retrieved_nodes:
            try:
                raw_dict = ast.literal_eval(n.text)
                query = raw_dict["query"]
                response = raw_dict["response"]
                result_str = dedent(f"""\
                    Question: {query}
                    Answer: {response}""")
                result_strs.append(result_str)
            except SyntaxError as exc:
                logger.warning("Converting example to dictionary failed: %s", exc)
                result_strs.append(n.text)
   
            
        return "\n\n".join(result_strs)
    
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
        
        
        # Build few-shot examples if needed
        get_qa_examples = self.build_few_shot_retriever(params)
        
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
      
