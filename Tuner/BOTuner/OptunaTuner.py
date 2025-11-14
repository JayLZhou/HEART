import optuna
import typing as T
import json
import traceback
from datetime import datetime, timezone
from Option.Config2 import Config
from Common.Logger import logger
from Tuner.BOTuner.HierarchicalTPE import HierarchicalTPESampler
from Tuner.BOTuner.BasicBOTuner import BasicBOTuner
from Pipeline.FlowBuild import FlowBuilder
# class SyftrEvaluationResult(EvaluationResult):
#     class Config:
#         arbitrary_types_allowed = True

#     qa_pair: T.Optional[core.QAPair] = Field(default=None, description="Q&A pair")
#     run_time: T.Optional[float] = Field(
#         default=np.nan, description="Flow completion time"
#     )
#     generation_exception: T.Optional[Exception] = Field(
#         default=None, description="Exception during generation"
#     )
#     evaluation_exception: T.Optional[Exception] = Field(
#         default=None, description="Exception during evaluation"
#     )
#     llm_call_data: T.List[LLMCallData] = Field(
#         default_factory=list,
#         description="Token counts and latencies for all LLM calls made during flow",
#     )
#     retriever_context_length: T.Optional[float] = Field(
#         default=None, description="Total length of retrieved contexts in tokens"
#     )
#     retriever_recall: T.Optional[float] = Field(
#         default=None, description="Retriever recall score in [0, 1]"
#     )

def set_trial(
    trial: optuna.trial.FrozenTrial | optuna.trial.Trial,
    config: Config | None = None,
    params: dict[str, str | bool | int | float] | None = None,
    is_seeding: bool | None = None,
    metrics: T.Dict[str, float] | None = None,
    flow_json: str | None = None,
):
    if params:
        flow_name = get_flow_name(str(params["rag_mode"]))
        trial.set_user_attr("flow_name", flow_name)
    if study_config:
        trial.set_user_attr("dataset", study_config.dataset.name)
    if is_seeding is not None:
        trial.set_user_attr("is_seeding", is_seeding)
    if flow_json:
        trial.set_user_attr("flow", flow_json)
    if metrics:
        set_metrics(trial, metrics)    

def _set_metric(trial: optuna.trial.BaseTrial, metric_name: str, score: T.Any):
    trial.set_user_attr("metric_" + metric_name, score)


def set_metrics(trial: optuna.trial.BaseTrial, metrics: T.Dict[str, float] | None):
    assert metrics, "No metrics provided"
    for metric_name, score in metrics.items():
        _set_metric(trial, metric_name, score)        


class OptunaTuner(BasicBOTuner):
    def __init__(self, config: Config, builder: FlowBuilder):
        self.config = config
        self.builder = builder
        self._tuner = self._create_tuner()

    def get_sampler(self) -> optuna.samplers.BaseSampler:
        if self.config.tuner.optimization.sampler == "tpe":
            return optuna.samplers.TPESampler(
                n_startup_trials=self.config.tuner.optimization.num_random_trials,
                constant_liar=True,
                multivariate=True,
            )
        elif self.config.tuner.optimization.sampler == "hierarchical":
            return HierarchicalTPESampler(
                constant_liar=True,
                n_startup_trials=self.config.optimization.num_random_trials,
            )
        else:
            raise ValueError("Invalid sampler")

    def _create_tuner(self) -> optuna.Study:
        """Get a study instance for optuna"""
        study_name = self.config.tuner.name

        
        if self.config.tuner.reuse_study:
            logger.info(
                "Reusing study '%s' or creating new one", study_name
            )
            if self.config.tuner.recreate_study:
                self.recreate_with_completed_trials(self.config, storage)
     

        sampler = self.get_sampler()
        study = optuna.create_study(
            study_name=study_name,
            directions=["maximize"],
            sampler=sampler,
        )
        # self.save_config(study, self.study_config)
        return study



    def _evaluate(
        self,
        params: T.Dict,
    ) -> T.Tuple[float, float, T.Dict[str, float | str]]:
        flow_start = datetime.now(timezone.utc).timestamp()
        logger.info("Evaluating flow with config: %s", params)

      
        results: T.Dict[str, T.Any] = self.evaluator.eval_dataset(
            study_config=self.study_config,
            dataset_iter=self.study_config.dataset,
            flow=flow,
            evaluation_mode=self.study_config.evaluation.mode,
        )
        import pdb
        pdb.set_trace()
        obj = results[self.study_config.optimization.objective_1_name]
        return obj, results

    def save_config(self, study: optuna.Study, config: Config):
        """Save study config to database"""
        attrs = config.model_dump(mode="json")
        logger.info("Saving study config of %s to the database", study.study_name)
        for attr, value in attrs.items():
            study.set_user_attr(attr, value)


    



    def __call__(self):
        trial = self._tuner.ask()
        search_space = self.config.tuner.search_space
        params: dict[str, str | bool | int | float]
        for i in range(self.config.tuner.optimization.num_retries_unique_params):
        
            params = search_space.sample(trial, self.config.tuner.tuner_params)
            
            if not self.config.tuner.optimization.skip_existing:
                logger.info("Using generated parameter combination without check")
                break
            # if not self._trial_exists(self.config.tuner.name, params):
            #     logger.info(
            #         "Found novel parameter combination after %i retries: %s",
            #         i,
            #         str(params),
            #     )
            #     break
        try:
         
            self.builder.build_flow(params, self.config)
            obj, metrics, flow_json = self._evaluate(params)
        except Exception as ex:
            logger.exception("Objective had an unhandled exception: %s", ex)
            metrics = {
                "failed": True,
                "exception_message": str(ex),
                "exception_stacktrace": traceback.format_exc(),
                "exception_class": ex.__class__.__name__,
            }
            flow_json = json.dumps(params)
            raise ex
        finally:
            set_trial(
                trial=trial,
                study_config=self.study_config,
                params=params,
                is_seeding=False,
                metrics=metrics,
                flow_json=flow_json,
            )
        self._tuner.tell(trial, [obj])

        return obj

   
 

    def _trial_exists(self,
    study_name: str,
    params: T.Dict[str, T.Any],
    storage: str) -> bool:
        storage = storage or self.study_config.database.get_optuna_storage()
        logger.debug("Loading '%s' from storage: %s", study_name, storage)
        study = optuna.load_study(study_name=study_name, storage=storage)
        for trial in study.get_trials():
            if params == trial.params:
                return True
        return False

# # 兼容性函数包装器，用于向后兼容
# def get_study(study_config: StudyConfig) -> optuna.Study:
#     """兼容性函数包装器，用于向后兼容"""
#     tuner = OptunaTuner(study_config)
#     return tuner.get_study()


# def objective(
#     trial: optuna.Trial,
#     study_config: StudyConfig,
#     components: T.List[str],
# ) -> T.Tuple[float, float]:
#     """兼容性函数包装器，用于向后兼容"""
#     tuner = OptunaTuner(study_config)
#     return tuner.objective(trial, components)


# def async_eval(
#     items: T.List[core.QAPair],
#     flow: Flow,
#     study_config: T.Union[StudyConfig, AgentStudyConfig],
#     evaluators: T.Sequence[BaseEvaluator],
#     rate_limiter: AsyncLimiter,
#     pruner: ParetoPruner | None = None,
#     cost_pruner: CostPruner | None = None,
#     timeout_pruner: RuntimePruner | None = None,
#     raise_on_exception: bool | None = EVAL__RAISE_ON_EXCEPTION,
# ) -> T.Tuple[T.List[SyftrEvaluationResult], T.Optional[str]]:
#     """Evaluate Q&A items asynchronously with an evaluator chosen at random for each pair."""
#     return _async_eval_runner(
#         _aeval_pair,
#         items,
#         flow,
#         evaluators,
#         study_config,
#         rate_limiter,
#         pruner=pruner,
#         timeout_pruner=timeout_pruner,
#         raise_on_exception=raise_on_exception,
#         cost_pruner=cost_pruner,
#     )
# # Todo
# def _async_eval_runner(
#     pair_eval_runner: T.Callable,
#     items: T.List[core.QAPair],
#     flow: Flow,
#     evaluators: T.Sequence[BaseEvaluator],
#     study_config: T.Union[StudyConfig, AgentStudyConfig],
#     rate_limiter: AsyncLimiter,
#     raise_on_exception: bool | None = EVAL__RAISE_ON_EXCEPTION,
#     pruner: ParetoPruner | None = None,
#     timeout_pruner: RuntimePruner | None = None,
#     cost_pruner: CostPruner | None = None,
# ) -> T.Tuple[T.List[SyftrEvaluationResult], T.Optional[str]]:
#     """Evaluate Q&A items asynchronously using provided pair_eval_runner."""

#     try:
#         loop = asyncio.get_event_loop()
#     except RuntimeError:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)

#     prune_reason = None
#     results = []
#     num_batches = math.ceil(len(items) / study_config.optimization.num_eval_batch)
#     for i, batch in enumerate(
#         itertools.batched(items, study_config.optimization.num_eval_batch)
#     ):
#         batch_result = loop.run_until_complete(
#             _aeval_all_pair_runner(
#                 pair_eval_runner,
#                 list(batch),
#                 flow,
#                 evaluators,
#                 study_config.timeouts.single_eval_timeout,
#                 rate_limiter,
#                 raise_on_exception,
#             )
#         )
#         results.extend(batch_result)
#         run_times = [
#             r.run_time for r in results if r and r.run_time and not np.isnan(r.run_time)
#         ]
#         # Compute stats and pruners if we have successful evals
#         # or, if we are over the max fail rate, proceed to calculate_metrics, which will
#         # error out if we don't have any successful trials.
#         # max_eval_failure_rate only applies if there are zero successful evals so far
#         if (
#             run_times
#             or i / num_batches > study_config.optimization.max_eval_failure_rate
#         ):
#             current_metrics = calculate_metrics(results, study_config)

#             log.info(
#                 "Finished evaluation batch %s/%s with %s QA pairs. Metrics: %s, Flow: %s",
#                 i + 1,
#                 num_batches,
#                 len(batch),
#                 {
#                     current_metrics["objective_1_name"]: current_metrics["obj1_value"],
#                     current_metrics["objective_2_name"]: current_metrics["obj2_value"],
#                     "num_errors": current_metrics["num_errors"],
#                 },
#                 flow,
#             )

#     return results, prune_reason

# async def _aeval_all_pair_runner(
#     pair_eval_runner: T.Callable,
#     dataset: T.List[core.QAPair],
#     flow: Flow,
#     evaluators: T.Sequence[BaseEvaluator],
#     eval_timeout: int,
#     rate_limiter: AsyncLimiter,
#     raise_on_exception: bool | None = EVAL__RAISE_ON_EXCEPTION,
# ) -> T.List[SyftrEvaluationResult]:
#     """Helper function to run multiple pair_eval_runners in parallel."""

#     tasks = []
#     for pair in dataset:
#         tasks.append(
#             asyncio.create_task( # value error: Calculated available context size -4901 was not non-negative.
#                 pair_eval_runner(
#                     pair, flow, evaluators, rate_limiter, raise_on_exception
#                 )
#             )
#         )
    
#     await asyncio.wait(tasks, timeout=eval_timeout)
#     all_results = []
#     for t in tasks:
#         try:
#             r = t.result()
#         except asyncio.exceptions.InvalidStateError as exc:
#             # Providing empty result for proper reporting.
#             exc.add_note(
#                 f"Eval of task {t} terminated due to timeout of {eval_timeout} seconds."
#             )
#             r = SyftrEvaluationResult(
#                 qa_pair=None,
#                 run_time=np.nan,
#                 generation_exception=exc,
#                 llm_call_data=[],
#             )
#         all_results.append(r)
#     return all_results

# async def _aeval_pair(
#     qa_pair: core.QAPair,
#     flow: Flow,
#     evaluators: T.Sequence[BaseEvaluator],
#     rate_limiter: AsyncLimiter,
#     raise_on_exception: bool | None = EVAL__RAISE_ON_EXCEPTION,
# ) -> SyftrEvaluationResult:
#     """Evaluate single Q&A item asynchronously."""
#     response, run_time, call_data, generation_exception = await agenerate_pair(
#         qa_pair, flow, rate_limiter
#     )
#     eval_result, evaluation_exception = None, None
#     if response:
#         evaluator = evaluators[0]
#         eval_result, evaluation_exception = await aevaluate_pair(
#             qa_pair, response, evaluator, rate_limiter, raise_on_exception
#         )
#     return SyftrEvaluationResult(
#         qa_pair=qa_pair,
#         run_time=run_time,
#         generation_exception=generation_exception,
#         evaluation_exception=evaluation_exception,
#         llm_call_data=call_data,
#         **(eval_result.model_dump() if eval_result else {}),
#     )
# async def aevaluate_pair(
#     qa_pair: core.QAPair,
#     response: CompletionResponse,
#     evaluator: BaseEvaluator,
#     rate_limiter: AsyncLimiter,
#     raise_on_exception: bool | None = EVAL__RAISE_ON_EXCEPTION,
# ) -> T.Tuple[EvaluationResult | None, Exception | None]:
#     """Evaluate a flow response asynchronously."""
#     async with rate_limiter:
#         return await exception_catcher(
#             func=evaluator.aevaluate,
#             return_values_on_exception=(None,),
#             raise_on_exception=raise_on_exception,
#             query=qa_pair.question,
#             response=response.text,
#             reference=qa_pair.answer,
#         )



# def _two_stage_decomposition_retrieval(self, queries: List[str]) -> List[List[NodeWithScore]]:
#         """两阶段高并发查询分解检索"""
#         # 🚀 阶段1：批量查询分解 (使用BatchLLMCaller默认并发数)
#         logger.info(f"🔄 阶段1：批量分解{len(queries)}个查询...")
#         from syftr.utils.batch_api_evaluator import get_batch_llm_caller
        
#         # 🔥 修复：从Flow配置中获取正确的查询分解模型名称
#         decomp_model_name = "Qwen2_5-7b"  # 默认值
#         if hasattr(self.flow, 'params') and self.flow.params:
#             decomp_model_name = self.flow.params.get('rag_query_decomposition_llm_name', 
#                                                    self.flow.params.get('query_decomposition_llm', 'Qwen2_5-7b'))
        
#         batch_caller = get_batch_llm_caller(model_name=decomp_model_name, max_workers=self.max_workers)
#         num_queries = self._get_num_queries()
        
#         decomp_results = batch_caller.batch_query_decomposition(queries, num_queries=num_queries)
#         logger.info(f"✅ [Fusion阶段1] 查询分解完成: 原始{len(queries)}个查询 → 期望{num_queries}个子查询/原始查询")
        
#         # 🚀 阶段2：构建扁平化检索映射
#         flatten_queries = []
#         query_mapping = []  # [(原始索引, 子查询索引), ...]
        
#         for orig_idx, sub_queries in enumerate(decomp_results):
#             effective_sub_queries = sub_queries if sub_queries else [queries[orig_idx]]
#             for sub_idx, sub_query in enumerate(effective_sub_queries):
#                 flatten_queries.append(sub_query)
#                 query_mapping.append((orig_idx, sub_idx))
        
#         logger.info(f"🔄 [Fusion阶段2] 查询映射构建完成: {len(queries)}个原始查询 → {len(flatten_queries)}个子查询")
#         logger.info(f"📊 [Fusion统计] 平均分解倍数: {len(flatten_queries)/len(queries):.1f}x")
        
#         logger.info(f"🔄 阶段2：高并发检索{len(flatten_queries)}个子查询...")
        
#         # 🚀 阶段3：高并发批量检索 (32并发)
#         MAX_RETRIEVAL_WORKERS = 32  # 大幅提升并发数
        
#         all_results = [[] for _ in flatten_queries]
#         with ThreadPoolExecutor(max_workers=MAX_RETRIEVAL_WORKERS) as executor:
#             future_to_index = {
#                 executor.submit(self._direct_retrieve_without_decomposition, QueryBundle(query)): i
#                 for i, query in enumerate(flatten_queries)
#             }
            
#             with tqdm(total=len(flatten_queries), desc="高并发子查询检索") as pbar:
#                 for future in as_completed(future_to_index):
#                     index = future_to_index[future]
#                     try:
#                         all_results[index] = future.result()
#                     except Exception as e:
#                         logger.warning(f"检索失败 query_idx={index}: {e}")
#                         all_results[index] = []
#                     finally:
#                         pbar.update(1)
        
#         # 🚀 阶段4：重组和融合结果
#         logger.info(f"🔄 [Fusion阶段3] 开始结果重组和融合...")
#         return self._regroup_and_fuse_results(all_results, query_mapping, len(queries))
    
#     def _regroup_and_fuse_results(self, all_results: List[List[NodeWithScore]], 
#                                  query_mapping: List[tuple], 
#                                  num_original_queries: int) -> List[List[NodeWithScore]]:
#         """重组：从扁平化结果重构到分组结果并融合"""
#         logger.info(f"🔗 [Fusion详细] 开始重组: {len(all_results)}个子查询结果 → {num_original_queries}个原始查询组")
        
#         # 按原始查询分组
#         grouped_results = [[] for _ in range(num_original_queries)]
        
#         for result_idx, (orig_idx, sub_idx) in enumerate(query_mapping):
#             grouped_results[orig_idx].append(all_results[result_idx])
        
#         logger.info(f"🔀 [Fusion详细] 重组完成，开始融合算法处理...")
        
#         # 融合：每组内的多个检索结果融合为1个
#         final_results = []
#         total_nodes_before = 0
#         total_nodes_after = 0
        
#         for group_idx, group_results in enumerate(grouped_results):
#             nodes_before = sum(len(nodes) for nodes in group_results)
#             fused = self._fuse_subquery_results(group_results)
#             nodes_after = len(fused)
            
#             total_nodes_before += nodes_before
#             total_nodes_after += nodes_after
#             final_results.append(fused)
        
#         logger.info(f"✅ [Fusion完成] 融合统计: {total_nodes_before}个节点 → {total_nodes_after}个节点 (压缩率: {total_nodes_after/total_nodes_before*100:.1f}%)")
        
#         return final_results

# def _fuse_subquery_results(self, sub_results: List[List[NodeWithScore]]) -> List[NodeWithScore]:
#         """高效融合算法：去重 + 相对分数融合 + Top-K选择"""
#         if not sub_results:
#             return []
        
#         # 统计融合前信息
#         total_input_nodes = sum(len(nodes) for nodes in sub_results)
#         logger.debug(f"🔀 [融合算法] 输入: {len(sub_results)}个子查询, 共{total_input_nodes}个节点")
        
#         # 使用字典实现O(1)去重
#         node_score_map = {}
        
#         for sub_idx, nodes in enumerate(sub_results):
#             # 子查询权重：后面的子查询权重略低
#             query_weight = 1.0 - (sub_idx * 0.1)
            
#             for node in nodes:
#                 node_id = getattr(node.node, 'node_id', hash(node.node.get_content()[:100]))
                
#                 # 相对分数融合：取最高分数
#                 weighted_score = node.score * query_weight
                
#                 if node_id not in node_score_map:
#                     node_score_map[node_id] = (node, weighted_score)
#                 else:
#                     existing_node, existing_score = node_score_map[node_id]
#                     if weighted_score > existing_score:
#                         node_score_map[node_id] = (node, weighted_score)
        
#         logger.debug(f"🔀 [融合算法] 去重后: {len(node_score_map)}个唯一节点 (去重率: {(1-len(node_score_map)/total_input_nodes)*100:.1f}%)")
        
#         # 按分数排序并返回Top-K
#         sorted_nodes = sorted(
#             [(node, score) for node, score in node_score_map.values()], 
#             key=lambda x: x[1], 
#             reverse=True
#         )
        
#         top_k = getattr(self.retriever, '_similarity_top_k', 20)
#         return [node for node, _ in sorted_nodes[:top_k]]        



# # Reranking 
# # 
# # # 🔄 阶段4: 批处理后处理（真正的批处理reranking）
#         rerank_phase_start = time.time()
#         logger.info("🚀 开始阶段4: 批处理后处理（真正的批处理reranking）")
        
#         # 统计检索结果
#         total_retrieved = sum(len(nodes) for nodes in retrieved_nodes_list)
#         avg_retrieved = total_retrieved / len(queries) if queries else 0
#         logger.info(f"📊 检索结果统计: 总文档数={total_retrieved}, 平均每查询={avg_retrieved:.1f}个文档")
        
#         # 4.1 批处理reranking（如果启用）
#         if self.parallel_reranker:
#             logger.info(f"🎯 启动批处理reranking: {self.parallel_reranker.reranker_name}...")
#             rerank_start = time.time()
#             processed_nodes_list = self.parallel_reranker.batch_rerank_all_queries(retrieved_nodes_list, queries)
#             rerank_time = time.time() - rerank_start
            
#             # 统计rerank结果
#             total_reranked = sum(len(nodes) for nodes in processed_nodes_list)
#             logger.info(f"✅ 批处理reranking完成: 耗时{rerank_time:.2f}s")
#             logger.info(f"📊 Rerank结果: 输出文档数={total_reranked}, 平均每查询={total_reranked/len(queries):.1f}个")
#         else:
#             # 如果没有reranker，直接返回原始结果（不截断）
#             logger.info("⚪ 未启用reranker，使用原始检索结果")
#             processed_nodes_list = retrieved_nodes_list
        
#         rerank_phase_time = time.time() - rerank_phase_start
#         logger.info(f"✅ 阶段4完成: 总耗时{rerank_phase_time:.2f}s")
        
#         # 4.2 构建最终结果
#         context_strs = self._batch_build_context_strings(processed_nodes_list)
#         final_prompts = self._batch_build_final_prompts(queries, context_strs)
        
#         processing_time = time.time() - start_time
#         # 成功计数现在基于最终生成的prompts，跳过的样本prompt为空
#         success_count = sum(1 for prompt in final_prompts if prompt.strip())
        
#         logger.info(f"🎉 批量RAG构建完成: 总耗时={processing_time:.3f}s, 实际处理={success_count}/{len(queries)}")
        
#         return BatchRAGPromptResult(
#             final_prompts=final_prompts,
#             queries=queries,
#             retrieved_nodes_list=retrieved_nodes_list,
#             processed_nodes_list=processed_nodes_list,
#             context_strs=context_strs,
#             processing_time=processing_time,
#             embedding_time=embedding_time,
#             retrieval_time=retrieval_time,
#             success_count=success_count,
#             failed_indices=[i for i, p in enumerate(final_prompts) if not p.strip()],
#             error_messages=[],
#             coreset_result=coreset_result_obj,
#             coreset_train_indices=coreset_train_indices,
#             query_embeddings=query_embeddings_np
#         )        


#   def _batch_build_context_strings(self, processed_nodes_list: List[List[NodeWithScore]]) -> List[str]:
#         """批量构建上下文字符串"""
#         context_strs = []
        
#         for nodes in processed_nodes_list:
#             context_parts = []
#             for i, node in enumerate(nodes):
#                 try:
#                     text = node.node.get_content()
#                     if text.strip():
#                         context_parts.append(f"Context {i+1}:\n{text.strip()}")
#                 except Exception as e:
#                     logger.warning(f"⚠️ 处理节点{i}失败: {e}")
#                     continue
            
#             context_str = "\n\n".join(context_parts)
#             context_strs.append(context_str)
        
#         return context_strs
    
#     def _batch_build_final_prompts(self, queries: List[str], context_strs: List[str]) -> List[str]:
#         """批量构建最终prompts"""
#         final_prompts = []
        
#         for query, context_str in zip(queries, context_strs):
#             try:
#                 # 使用prompt模板构建最终prompt
#                 if hasattr(self.prompt_template, 'format'):
#                     try:
#                         final_prompt = self.prompt_template.format(
#                             context_str=context_str,
#                             query_str=query
#                         )
#                     except (KeyError, TypeError):
#                         try:
#                             final_prompt = self.prompt_template.format(
#                                 context=context_str,
#                                 query=query
#                             )
#                         except (KeyError, TypeError):
#                             # 备用方案：简单字符串替换
#                             template_str = str(self.prompt_template)
#                             final_prompt = template_str.replace("{context_str}", context_str)
#                             final_prompt = final_prompt.replace("{query_str}", query)
#                 else:
#                     # 最简单的备用方案
#                     final_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
                
#                 final_prompts.append(final_prompt)
#             except Exception as e:
#                 logger.warning(f"⚠️ 构建prompt失败: {e}")
#                 final_prompts.append(f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:")
        
#         return final_prompts


# # Import reranker classes from factory
# from Rerank.RerankFactory import BatchReranker, build_reranker_postprocessor
