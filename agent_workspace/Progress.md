# Progress Log

## 2026-03-21

### Task

Read the LGBO paper in `lgbo/4356_Unleashing_LLMs_in_Bayesi.pdf` and align the paper's method with the implementation in `lgbo/`.

### Paper Summary

- The paper proposes LGBO, an LLM-guided Bayesian optimization framework for scientific discovery.
- Its main claim is that LLMs should not only be used for warm start or candidate generation.
- Instead, at every BO iteration, the LLM provides a structured preference:
  - `point`: `[point, [x1, ..., xd], confidence]`
  - `region`: `[region, [[lb1, ..., lbd], [ub1, ..., ubd]], confidence]`
- This preference is converted into a region-lifted preference that shifts the GP surrogate mean while leaving the covariance unchanged.
- The acquisition function still makes the final selection, so the LLM acts as continuous semantic guidance rather than as a direct optimizer.

### Paper -> Code Mapping

#### 1. LLM structured output

- Paper: the LLM must output either a point or a region with confidence.
- Code:
  - `lgbo/prompt.py` defines the structured prompt and output rules.
  - `lgbo/exp.py` calls the model with `call_chat(...)`.
  - `lgbo/prompt.py` parses the reply via `_parse_assistant(...)`.

#### 2. Continuous guidance in every round

- Paper: the LLM is queried at every optimization round, not only once.
- Code:
  - `lgbo/exp.py` implements the main loop in `run_case(...)`.
  - Each round:
    - build current user prompt
    - call LLM
    - parse point/region suggestion
    - convert suggestion to BO plan
    - propose BO points
    - evaluate and append history
    - build the next-round prompt with history and reasoning trace

#### 3. Convert point/region preference into a BO plan

- Paper: convert LLM preference into a tractable lifted preference with confidence-controlled strength.
- Code:
  - `lgbo/exp.py` uses `build_expert_input_from_parsed(...)` to normalize parsed suggestions into:
    - `["point", normalized_point, confidence]`
    - `["region", [normalized_lb, normalized_ub], confidence]`
  - `lgbo/decide.py` uses:
    - `parse_expert_input(...)`
    - `confidence_to_delta(...)`
    - `decide_preference(...)`
    - `decide_preference_tilt_from_expert(...)`
- Important implementation detail:
  - point suggestions are converted into a small region-like tilt via `decide_preference_tilt_from_expert(...)`, rather than staying as a pure point prior.
  - region suggestions may become:
    - `point`
    - `region-soft`
    - `region`
    depending on effective coverage under the Sobol grid.

#### 4. Region-lifted preference as surrogate mean shift

- Paper: the preference changes the surrogate mean, not the covariance.
- Code:
  - `lgbo/prior.py` contains `LinearExponentialRegionalMeanTiltPlugAndPlay`.
  - `lgbo/prior.py` also contains `TiltedModel`.
  - `lgbo/boo.py` applies the region preference by:
    - creating the regional tilt object
    - fitting lambda from confidence-derived delta
    - wrapping the base model with `TiltedModel`
    - evaluating acquisition on the tilted model

#### 5. Final next-point selection is still done by BO acquisition

- Paper: acquisition remains responsible for final experiment selection.
- Code:
  - `lgbo/boo.py` implements `propose_points_from_plan(...)`.
  - In region mode, it creates a tilted model and then uses `qLogExpectedImprovement`.
  - In point or value mode, it uses location/value priors through the weighted BO sampler.
  - The actual candidate selection still comes from BO acquisition logic rather than directly from the LLM output.

### Important Observations

- The paper describes both warm start and continuous guidance.
- The current `lgbo` toy loop in `exp.py` uses Sobol initialization (`N_INIT`, `SobolEngine`) rather than LLM-generated initial points.
- Therefore, the current implementation preserves the core idea of continuous LLM guidance, but does not fully mirror the paper's warm-start description in the toy setting.
- The implementation is organized roughly as:
  - `exp.py`: orchestration loop
  - `prompt.py`: prompt construction and parsing
  - `decide.py`: convert expert output into a tractable preference plan
  - `boo.py`: inject the plan into BO and propose actual points
  - `prior.py`: mean-tilt machinery

### Current Conclusion

- I now have a usable alignment between the paper method and the current `lgbo` implementation.
- Next useful step: compare this implementation with `HEART/Tuner/BOTuner/LLMBO.py` and identify which parts of the paper logic have already been ported and which are still missing or mismatched.

### Correction Of Integration Target

- Important correction from the user:
  - `HEART/Tuner/BOTuner/LLMBO.py` is a different feature.
  - The actual task is not to modify that file into LGBO.
  - Instead, the goal is to implement a new `LGBO.py` inside `HEART/Tuner/BOTuner/`.
- `LLMBO.py` should be treated as a reference for how HEART integrates a custom BO method with Optuna.

### HEART BO Integration Findings

#### 1. Where the BO tuner is instantiated

- `HEART/Tuner/TunerFactory.py`
  - BO tuners are created through `OptunaTuner`.
  - Therefore, any new LGBO support must ultimately be reachable through `OptunaTuner`.

#### 2. Where sampler selection happens

- `HEART/Tuner/BOTuner/OptunaTuner.py`
  - `get_sampler()` switches on `self.config.tuner.optimization.sampler`
  - currently supports:
    - `tpe`
    - `hierarchical`
    - `llmbo`
- This means a future LGBO integration will likely require:
  - a new sampler option such as `lgbo`
  - importing `LGBOSampler`
  - returning that sampler from `get_sampler()`

#### 3. How custom sampler logic is actually used

- `OptunaTuner.__call__()` does not rely only on vanilla Optuna sampling.
- There is a special path for `llmbo`:
  - call `trial = self._tuner.ask()`
  - create a fresh sampler with `self.get_sampler()`
  - infer search space
  - load the study explicitly
  - call `sampler.sample_relative(study, trial, search_space)`
  - use the returned params to build and evaluate the flow
- This is important because it shows HEART is already willing to bypass pure Optuna internal sampling and directly invoke a custom sampler.
- A new LGBO integration can follow the same pattern.

#### 4. What `LLMBO.py` is useful for

- `HEART/Tuner/BOTuner/LLMBO.py` is useful mainly as an Optuna integration template.
- Key reusable ideas from `LLMBO.py`:
  - implement a sampler-like class (`LLMBOSampler`)
  - expose `infer_relative_search_space(...)`
  - expose `sample_relative(...)`
  - extract completed Optuna trials from the study
  - reconstruct observed configs / objective values from trial history
  - convert Optuna distributions into an internal hyperparameter schema
  - return a parameter dictionary that HEART can pass into `build_flow(...)`
- This is more important for HEART integration than the `lgbo` reference code, because the original LGBO repo does not use Optuna.

### Updated Working Plan

- Do not modify `HEART/Tuner/BOTuner/LLMBO.py` into LGBO.
- Use `lgbo/` as the algorithm source.
- Use `HEART/Tuner/BOTuner/LLMBO.py` and `OptunaTuner.py` as the integration reference.
- The next concrete step should be:
  - design `HEART/Tuner/BOTuner/LGBO.py` as an Optuna-compatible sampler module
  - identify which pieces of `lgbo/` need adaptation into the Optuna trial-history model used by HEART
  - identify what extra prompt / preference / mean-tilt machinery must be ported from `lgbo/`

### Proposed LGBO Design In HEART

This section defines how LGBO should work inside HEART before implementation.

#### A. Overall workflow

The future `HEART/Tuner/BOTuner/LGBO.py` should follow this round-level workflow:

1. Optuna creates or exposes the current study/trial.
2. LGBO reads the search space from Optuna distributions.
3. LGBO extracts completed trial history from the study.
4. LGBO converts trial history into:
   - observed configurations
   - observed objective values
   - optional textual trajectory summary for prompting
5. LGBO builds a structured prompt from:
   - query/task context
   - search-space description
   - recent optimization trajectory
   - previous-round reasoning trace
6. LLM returns a structured preference:
   - point
   - region
   - confidence
7. LGBO converts that preference into a BO plan:
   - point prior
   - region-soft tilt
   - region tilt
   - none/fallback
8. LGBO uses BO acquisition on the updated surrogate to generate candidate points.
9. LGBO selects one final parameter assignment compatible with the Optuna search space.
10. HEART evaluates the flow and writes result metrics back into the Optuna trial.
11. LGBO uses the stored trajectory on the next round.

#### B. What to extract from HEART / Optuna

LGBO should not try to reconstruct everything from files on disk first. The primary source of truth should be the Optuna study.

From each completed trial, LGBO should extract:

- `trial.params`
  - the actual sampled hyperparameter configuration
- `trial.value` or `trial.values`
  - the optimization objective result
- `trial.user_attrs["flow"]`
  - the serialized configuration used to build the flow
- `trial.user_attrs["query"]`
  - the current query/task payload
- `trial.user_attrs["metric_*"]`
  - auxiliary evaluation metrics if useful for summarization

This gives three layers of trajectory:

1. Numeric trajectory
   - params + objective values
2. Configuration trajectory
   - full flow JSON if needed for exact reproduction
3. Semantic trajectory
   - query + selected metrics + prior reasoning summary

#### C. How to store trajectory

There are two different storage needs and they should not be mixed.

##### 1. Long-term canonical storage

Use existing Optuna storage as the canonical source of truth:

- parameters live in `trial.params`
- raw config snapshot lives in `trial.user_attrs["flow"]`
- query/task snapshot lives in `trial.user_attrs["query"]`
- evaluation metrics live in `trial.user_attrs["metric_*"]`

This avoids inventing a second database.

##### 2. LGBO-specific short-term trajectory state

LGBO also needs a lightweight state for prompt continuity across rounds. This should be stored in the same Optuna study/trial metadata layer, not in an external memory system first.

Recommended storage:

- On each trial:
  - `trial.user_attrs["lgbo_preference_raw"]`
  - `trial.user_attrs["lgbo_preference_parsed"]`
  - `trial.user_attrs["lgbo_plan"]`
  - `trial.user_attrs["lgbo_reasoning"]`
- On the study:
  - optional study-level attr for latest reasoning summary if needed

Reason:

- the paper-style workflow uses previous-round reasoning
- this is not present in default Optuna history
- storing it in user attrs keeps everything reproducible and query-local

#### D. How to generate preference

Preference generation should be an LLM-facing module, separate from BO math.

Input to preference generation:

- query text
- task metadata
- parameter names and bounds
- objective direction and metric name
- top historical trials
- recent failed / low-quality regions if useful
- previous reasoning trace

Output format should stay close to the paper:

- `[point, [x1, ..., xd], confidence]`
- `[region, [[lb1, ..., lbd], [ub1, ..., ubd]], confidence]`

Inside HEART, this module should be isolated behind something like:

- `build_lgbo_prompt(...)`
- `call_lgbo_llm(...)`
- `parse_lgbo_preference(...)`

This makes the prompting layer replaceable without touching BO logic.

#### E. How to generate candidates

Candidate generation should be the BO-facing module, separate from prompt generation.

Input to candidate generation:

- observed configs
- observed objective values
- Optuna-derived search space
- parsed preference / tilt plan

Output:

- one or more candidate parameter dictionaries

The intended logic is:

1. fit or construct the surrogate from observed trials
2. if no valid preference is available, fall back to pure BO
3. if a point preference is available:
   - convert to local preference region or point prior
4. if a region preference is available:
   - convert to region lift / mean tilt
5. run acquisition on the updated surrogate
6. produce candidates inside the legal Optuna search space

This module should correspond conceptually to `propose_points_from_plan(...)` in `lgbo/`, but adapted to HEART's tabular/search-space setting.

### Repository Layout Update

- Moved the external LGBO reference repo into `HEART/lgbo/`.
- Moved `Progress.md` into `HEART/agent_workspace/Progress.md`.
- `agent_workspace/` is now the intended place for progress tracking and future agent-specific assets.

#### F. Recommended module split inside `HEART/Tuner/BOTuner/LGBO.py`

The new LGBO integration should be split into small, explicit components instead of one large file.

Recommended components:

1. `LGBOSampler`
   - Optuna-facing entry point
   - implements `infer_relative_search_space(...)`
   - implements `sample_relative(...)`

2. `LGBOHistoryAdapter`
   - extract completed trials from Optuna study
   - build `observed_configs`
   - build `observed_fvals`
   - collect previous LGBO reasoning / preference records

3. `LGBOPromptBuilder`
   - build structured prompt from query + history + search space

4. `LGBOPreferenceParser`
   - parse LLM output into normalized point/region/confidence objects

5. `LGBOPreferencePlanner`
   - convert parsed preference into a tractable BO plan
   - point / region-soft / region / none

6. `LGBOCandidateGenerator`
   - fit surrogate
   - apply preference plan
   - run acquisition
   - return final params

7. `LGBOTraceStore`
   - read/write LGBO-specific attrs into trial/study metadata

#### G. Minimal practical workflow for first implementation

To reduce risk, the first version of LGBO in HEART should be intentionally narrower than the paper.

Recommended V1:

1. use Optuna study history as the only trajectory source
2. store previous reasoning in trial user attrs
3. support only continuous/int parameters first
4. support point and axis-aligned region preference
5. if preference parsing fails, fall back to pure BO
6. return one final params dict per round

Defer for later:

- richer categorical handling
- multi-candidate batch execution
- warm-start from external similar studies
- more advanced study-level memory

### Current Design Decision

- Trajectory extraction source: Optuna study/trial history
- Trajectory storage source of truth: Optuna params + user_attrs
- Preference generation layer: separate LLM prompt/parse pipeline
- Candidate generation layer: separate BO/surrogate/acquisition pipeline
- Integration point: `OptunaTuner` via a new `LGBOSampler`

### HEART Search-Space Findings

I traced the actual optimization parameter definitions through:

- `HEART/main.py`
- `HEART/Option/Config2.py`
- `HEART/Config/TunerConfig.py`
- `HEART/Config/SearchSpace.py`
- `HEART/Config/RetrieverConfig.py`
- `HEART/Config/RerankConfig.py`
- `HEART/Pipeline/FlowBuild.py`

#### A. Current optimization parameters that really matter

The optimizer is currently centered around these components:

1. top-level response/template selection
   - `template_name`
   - `response_synthesizer_llm`

2. retriever-side parameters under `rag_retriever`
   - `method`
   - `top_k`
   - `query_decomposition_enabled`
   - `hybrid_bm25_weight` (relevant when method is hybrid)
   - `query_decomposition_llm_name` (relevant when decomposition is enabled)
   - `query_decomposition_num_queries` (relevant when decomposition is enabled)
   - `fusion_mode` (relevant when hybrid or decomposition is active)

3. reranker-side parameters under `reranker`
   - `reranker_name`
   - `reranker_top_k`
   - plus a derived runtime flag `reranker_enabled`

These are the actual knobs that affect `FlowBuilder.build_flow(...)` and `FlowBuilder.get_retriever(...)`.

#### B. Parameters that are present in config defaults but not actually wired cleanly

There are a few mismatches in the current repo:

1. `TunerConfig.tuner_params` default includes:
   - `template_name`
   - `response_synthesizer_llm`
   - `reranker`
   - `rag_retriever`
   - `reranker`
   - `sub_question`

2. But `SearchSpace.build_distributions(...)` only explicitly handles:
   - `template_name`
   - `response_synthesizer_llm`
   - all retriever distributions
   - reranker distributions only if `"reranker"` is in params

3. `sub_question` is not currently integrated into `SearchSpace.build_distributions(...)` or `SearchSpace.sample(...)`.

4. `QueryConfig` exists as a separate config object, but its distributions are not currently plugged into the main `SearchSpace`.

So, for practical purposes, the fixed optimization problem currently targets:

- template choice
- response synthesizer LLM choice
- retriever strategy and its retrieval/fusion/query-decomposition knobs
- reranker type and reranker top-k

#### C. Important implementation nuance

The effective Optuna parameter names used by custom samplers are flattened names, such as:

- `template_name`
- `response_synthesizer_llm`
- `rag_method`
- `rag_top_k`
- `rag_query_decomposition_enabled`
- `rag_hybrid_bm25_weight`
- `rag_query_decomposition_llm_name`
- `rag_query_decomposition_num_queries`
- `rag_fusion_mode`
- `reranker_name`
- `reranker_reranker_top_k`

Then `OptunaTuner.wrap_params(...)` reconstructs part of the nested runtime config by grouping:

- `rag_*` -> `rag_retriever`
- `reranker_*` -> `reranker`

This flatten-then-wrap convention is what a future `LGBOSampler` should follow.

#### D. Current answer to the user's question

The optimization parameters are not arbitrary per query; they are fixed by the HEART search-space code.

The current practical parameter family is:

- prompt template
- response synthesizer model
- retriever method
- retriever top-k
- query decomposition on/off
- query decomposition LLM
- query decomposition number of generated subqueries
- hybrid BM25 weight
- fusion mode
- reranker model
- reranker top-k

This is the parameter set that future LGBO should reason over first.

### Development Constraint Agreed

- Avoid relying on actually running the full HEART system while modifying code.
- Prefer implementation and verification strategies that do not require installing the full dependency stack.
- Favor lightweight unit tests for pure logic and adapter behavior.
- Use mocks or fakes for Optuna studies, trials, LLM calls, and storage where possible.
- Defer full end-to-end execution until the code structure is stable and environment setup is intentionally approved.

### Implementation Started

I started the LGBO implementation with a dependency-light V1 skeleton.

#### Added files

- `HEART/Tuner/BOTuner/LGBO.py`
- `HEART/Tuner/BOTuner/lgbo_components/__init__.py`
- `HEART/Tuner/BOTuner/lgbo_components/history.py`
- `HEART/Tuner/BOTuner/lgbo_components/search_space.py`
- `HEART/Tuner/BOTuner/lgbo_components/preference.py`
- `HEART/Tuner/BOTuner/lgbo_components/trace_store.py`
- `HEART/Tuner/BOTuner/lgbo_components/candidate.py`
- `HEART/Prompt/LGBOPrompt.py`
- `HEART/tests/test_lgbo_components.py`

#### Updated files

- `HEART/Config/OptimizationConfig.py`
  - added `lgbo` as a valid sampler option
- `HEART/Tuner/BOTuner/OptunaTuner.py`
  - imported `LGBOSampler`
  - wired `lgbo` into `get_sampler()`
  - routed `lgbo` through the same manual custom-sampler path as `llmbo`

#### What the current V1 implementation already does

- exposes a new `LGBOSampler`
- flattens and reads the Optuna search space
- filters the search space to numeric-only parameters (`int` / `float`)
- extracts completed trial history from the study
- provides lightweight trace storage helpers for:
  - raw preference
  - parsed preference
  - plan
  - reasoning
- provides a paper-style preference parser for:
  - point
  - region
- provides a lightweight candidate generator that currently:
  - uses point preference directly
  - uses region midpoint for region preference
  - otherwise falls back to a safe numeric-only candidate policy

#### Important limitation of the current code

- This is not yet the full LGBO algorithm from the paper.
- The current `LGBOSampler` is a structural integration skeleton plus pure-logic helpers.
- The current candidate generation is intentionally lightweight so it can be tested without the full runtime stack.
- Full preference-driven prompting and surrogate/acquisition integration still need to be added on top of this skeleton.

#### Lightweight verification completed

- Ran `python -m py_compile` on the newly added LGBO-related files.
- This confirmed the new files are syntactically valid without requiring the full HEART environment to run.

## 2026-03-25

### Environment Update

- Activated the `graphrag` conda environment for follow-up verification work.
- Verified the active environment name as `graphrag`.
- Verified Python availability after activation:
  - `which python` -> `/home/yingli/bin/python`
  - `python --version` -> `Python 3.11.10`

### Current Verification Note

- The repository now reads API keys from a local `.env` file through environment-variable expansion in YAML config loading.
- A config smoke test succeeded after the `.env` change.
- `Provider/test_provider.py` still needs to be re-run inside the activated environment with the correct project import path.

### LGBO Gap Review And Implementation

- Read `lgbo/` paper notes and reference code again, then compared them against the current HEART-side `LGBOSampler` skeleton.
- Wrote the comparison and implementation checklist to `agent_workspace/plan.md`.

#### Main gaps identified

- `LGBOSampler.sample_relative(...)` had no prompt -> LLM -> parse -> plan loop.
- The preference parser only supported exact bracket literals and could not extract realistic `Thinking` / `Final Answer` output.
- The planner did not reflect the paper-inspired idea that point guidance should act like a compact local region preference.
- Candidate generation was objective-agnostic on fallback and simply reused the latest observation.
- Trial trace storage did not persist real raw responses, parsed preferences, or reasoning.

#### Implementation completed in this pass

- Updated `Prompt/LGBOPrompt.py`
  - added a reusable numeric LGBO system prompt
  - improved the numeric prompt builder to include:
    - task/query text
    - parameter bounds and order
    - completed-trial history
    - previous reasoning
- Updated `Tuner/BOTuner/lgbo_components/history.py`
  - added best-observation selection
  - added latest-query extraction from study/trial attrs
  - added compact history-line rendering for prompt construction
- Updated `Tuner/BOTuner/lgbo_components/preference.py`
  - added tolerant extraction of `Thinking` and `Final Answer`
  - added metadata-aware parsing
  - added point -> `region-soft` planning so local point advice becomes a compact soft neighborhood
- Updated `Tuner/BOTuner/lgbo_components/candidate.py`
  - added `region-soft` candidate handling
  - made fallback candidate selection objective-aware using the best completed observation
- Updated `Tuner/BOTuner/LGBO.py`
  - kept the external sampler interface unchanged
  - wired in:
    - query/history collection
    - prompt construction
    - LLM provider call via existing provider stack
    - parsing and planning
    - trace writing for raw output / parsed preference / final plan / reasoning
  - preserved graceful fallback behavior when LLM calling or parsing fails

#### Test updates

- Expanded `tests/test_lgbo_components.py` to cover:
  - realistic prompt parsing with `Thinking` + `Final Answer`
  - point -> `region-soft` planning
  - objective-aware fallback candidate selection
  - prompt construction with history and prior reasoning
- Adjusted the test loader so these component tests can run without importing the full heavyweight tuner stack.

#### Verification completed

- Ran:
  - `python tests/test_lgbo_components.py`
  - `python -m py_compile Prompt/LGBOPrompt.py Tuner/BOTuner/LGBO.py Tuner/BOTuner/lgbo_components/history.py Tuner/BOTuner/lgbo_components/preference.py Tuner/BOTuner/lgbo_components/candidate.py tests/test_lgbo_components.py`
- Result:
  - component tests passed
  - modified LGBO files compiled successfully

#### Remaining limitation after this pass

- HEART-side LGBO now has a real prompt-guided loop and better lightweight planning, but it still does not port the full BoTorch region-tilt posterior machinery from `lgbo/prior.py` / `lgbo/boo.py`.
- The current HEART implementation remains a dependency-light approximation designed to preserve interfaces and make incremental integration safer.

### External Dataset Preparation

- Cloned `https://github.com/ianliuwd/HippoRAG2.git` to the same directory level as `HEART`:
  - `/home/yingli/Youran/HippoRAG2`
- Confirmed the external dataset files under:
  - `/home/yingli/Youran/HippoRAG2/reproduce/dataset`

### HEART Dataset Subset Update

- Initially created a `hotpotqa` file with 100 text lines after misreading the request.
- Corrected this immediately after clarification from the user.
- Removed the incorrect line-based fragment file.
- Created a proper 100-record JSON subset at:
  - `HEART/dataset/hotpotqa_100_records.json`
- Current subset details:
  - source: `HippoRAG2/reproduce/dataset/hotpotqa.json`
  - selection rule: first 100 records
  - verified record count: 100

### Pipeline Bring-Up Notes

- Converted `HEART/dataset/hotpotqa_100_records.json` into HEART's expected runtime dataset layout:
  - `HEART/dataset/hotpotqa_100/Question.json`
  - `HEART/dataset/hotpotqa_100/Corpus.json`
- Verified:
  - `Question.json` contains 100 records
  - `Corpus.json` contains 994 deduplicated context documents
- Installed missing runtime dependencies in the `graphrag` environment to unblock indexing / retrieval / evaluation:
  - `llama-index-*`
  - `faiss-cpu`
  - `flashrank`
  - `PyStemmer`
  - `modelscope`
  - `rouge-score`
  - `mauve-text`
  - `langchain`

### LangChain Version Decision

- Checked installed versions:
  - `langchain==1.2.13`
  - `langchain-core==1.2.22`
- Verified that in this environment:
  - `FewShotPromptTemplate` and `PromptTemplate` are available from `langchain_core.prompts`
  - they are not available from `langchain`
  - `langchain.prompts` is also absent
- Searched the repository and found only one LangChain usage site:
  - `Tuner/BOTuner/LLMBO.py`
- Decision:
  - align the import to the actually installed modern package layout
  - use `from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate`
  - do not keep multi-version fallback imports

### Pipeline Smoke Results So Far

- Fixed-parameter RAG smoke run on `hotpotqa_100` succeeded end-to-end:
  - dataset loading
  - chunking
  - dense indexing
  - BM25 indexing
  - retrieval
  - flashrank reranking
  - response generation
  - single-query evaluation
- Observed smoke response on the first example was incorrect, so metrics were all zero for that sample, but the full runtime chain executed successfully.
- Tuner-path smoke then exposed environment / compatibility issues in `LLMBO.py`, which are being resolved incrementally.

### LangChain Import Cleanup

- The user requested not to keep multi-version fallback imports.
- After checking the actual environment, I updated `Tuner/BOTuner/LLMBO.py` to use a single explicit import:
  - `from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate`
- Reason:
  - the active environment uses `langchain==1.2.13`
  - that version does not expose these prompt classes from `langchain`
  - and does not provide `langchain.prompts`

### Runtime Bring-Up Results

#### 1. Fixed-parameter full RAG smoke run

- Ran a fixed-parameter smoke pipeline on the converted `hotpotqa_100` dataset using:
  - HF embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
  - FAISS dense index
  - BM25 sparse index
  - FlashRank reranker
  - `Qwen/Qwen3-8B` as the response synthesizer
- Result:
  - indexing succeeded
  - retrieval succeeded
  - reranking succeeded
  - generation succeeded
  - single-query evaluation succeeded
- The returned answer for the first sample was incorrect, but the runtime chain itself completed successfully.

#### 2. Tuner-path smoke run

- Ran a 1-query / 1-trial Optuna smoke path on the same dataset.
- This advanced far enough to:
  - build both indexes
  - create the Optuna study
  - sample a trial configuration
  - enter `OptunaTuner.__call__()`
- Current blockers discovered:

1. LLM pool / search-space mismatch
   - the sampled config chose `response_synthesizer_llm='gpt-4o'`
   - but the active `llms` pool only contained `Qwen/Qwen3-8B`
   - this caused `ContextMixin.get_llm()` to fail with:
     - `Model 'gpt-4o' not found in LLM pool`

2. Trial metric persistence bug on exception path
   - after the model-pool mismatch, `OptunaTuner._set_trial()` tried to write every metric value as `score * 0.01`
   - some exception metrics are strings, so this raised:
     - `TypeError: can't multiply sequence by non-int of type 'float'`

### Current Practical Conclusion

- The non-tuner pipeline is now runnable on the prepared `hotpotqa_100` dataset.
- The tuner path is close, but still blocked by:
  - config/search-space inconsistency around allowed LLM names
  - exception-metric handling in `OptunaTuner._set_trial()`

### Config Pool Narrowing

- Updated `Option/Config2.yaml` so that only the currently used Qwen model remains active in the `llms` pool:
  - `Qwen/Qwen3-8B`
- Commented out the other configured entries instead of deleting them:
  - `TA/meta-llama/Llama-3-8b-chat-hf`
  - `gpt-4o-mini`
  - `gpt-4o`

### Tuner Bring-Up Follow-up Fixes

- Fixed the earlier tuner exception-path blockers in `Tuner/BOTuner/OptunaTuner.py`:
  - align sampled LLM names to the currently configured `config.llms` pool before running the tuner
  - apply the same alignment to the nested tuner search-space retriever query-decomposition LLM list
  - stop scaling non-numeric exception metrics by `0.01` in `_set_trial()`
- Removed active `pdb.set_trace()` breakpoints from `Tuner/BOTuner/LLMBO.py` so the sampler no longer halts during async generation.
- Updated `Storage/QueryStorage.py` to support the prepared dataset's query schema:
  - accept either `id` or `_id`
  - convert query keys to strings before inserting into FAISS-backed storage
  - fall back to either metadata key on retrieval

### Latest Smoke Result

- Re-ran `hotpotqa_100` with a single-query / single-trial `llmbo` smoke using `Qwen/Qwen3-8B` and HF CPU embeddings.
- Result:
  - the tuner path now completed successfully with exit code `0`
  - a sampled trial used `response_synthesizer_llm='Qwen/Qwen3-8B'`
  - the sampled `rag_query_decomposition_llm_name` also correctly resolved to `Qwen/Qwen3-8B`
  - final returned metrics were:
    - `accuracy: 0.0`
    - `f1: 12.5`
    - `precision: 9.0909`
    - `recall: 20.0`
    - `em: 0.0`

### Residual Runtime Risk

- During the successful smoke run, the `upr` reranker logged:
  - `'str' object has no attribute 'question'`
- The pipeline continued and produced final metrics, so this is no longer a bring-up blocker, but it remains a correctness / robustness issue worth fixing before larger-scale runs.

### UPR Reranker Fix

- Fixed `Rerank/Upr.py` so it no longer assumes `document.question` is always an object with a nested `.question` attribute.
- Added a small normalization helper so both forms are supported:
  - plain string question
  - object-style question with a `.question` field
- Updated both UPR code paths to use the normalized question text:
  - T5 path
  - GPT path

### UPR Verification

- Ran a fixed single-query flow with:
  - sparse retrieval
  - `upr` reranker
  - `Qwen/Qwen3-8B` response synthesizer
- Result:
  - reranker initialized successfully
  - reranking completed successfully
  - no `'str' object has no attribute 'question'` error was emitted
  - the flow returned a final response and exited with code `0`

### Multi-record LLmBO Smoke

- Ran a small batch smoke on the first `3` records of `hotpotqa_100`.
- Setup:
  - build indexes once
  - run `llmbo` with `1 trial` per record
  - keep `Qwen/Qwen3-8B` as the only configured LLM
- Result summary:
  - record `0`: success
    - metrics: `accuracy=0.0`, `f1=0.0`, `precision=0.0`, `recall=0.0`, `em=0.0`
  - record `1`: failed
  - record `2`: failed

### New LLmBO Cold-start Bug Found

- The failing records exposed a new bring-up bug in `Tuner/BOTuner/LLMBO.py`.
- Symptom:
  - `Observation: Empty DataFrame`
  - followed by:
    - `ValueError: zero-size array to reduction operation maximum which has no identity`
- Immediate cause:
  - `get_candidate_points(...)` computes a numeric range using `np.max(observed_fvals.values)` and `np.min(observed_fvals.values)` even when there are no prior observations.
- Practical conclusion:
  - current `llmbo` path is not yet robust for all first-record / no-history situations
  - single-record success is possible, but multi-record runs reveal a remaining cold-start stability gap

### LGBO Runtime Bring-up Resumed

- Confirmed the intended debugging target is `lgbo`, not `llmbo`.
- Verified the recent drift happened because the smoke script was still explicitly setting:
  - `opt.tuner.optimization.sampler = 'llmbo'`
- Switched the runtime smoke path back to `lgbo` without changing public interfaces.

### LGBO Optuna Compatibility Fix

- The first real `lgbo` smoke failed before sampler logic ran because Optuna's `Study.ask()` called a sampler hook that `LGBOSampler` did not implement:
  - `AttributeError: 'LGBOSampler' object has no attribute 'before_trial'`
- Fixed `Tuner/BOTuner/LGBO.py` by adding minimal Optuna-compatible hook methods:
  - `before_trial(...)`
  - `after_trial(...)`
- Kept them as no-op implementations, matching the current lightweight integration style used by the custom sampler path.

### LGBO Smoke Result

- Re-ran a real `lgbo` smoke on `hotpotqa_100` with:
  - `1 query`
  - `1 trial`
  - HF CPU embeddings
  - `Qwen/Qwen3-8B` as the configured model pool
- Result:
  - tuner creation succeeded
  - custom `lgbo` sampler executed successfully
  - a full trial completed end-to-end with exit code `0`
  - sampled params included:
    - `response_synthesizer_llm='Qwen/Qwen3-8B'`
    - `rag_query_decomposition_llm_name='Qwen/Qwen3-8B'`
  - final metrics were:
    - `accuracy: 0.0`
    - `f1: 0.0`
    - `precision: 0.0`
    - `recall: 0.0`
    - `em: 0.0`

### Current LGBO Status

- `lgbo` is now runnable through HEART's Optuna path on the prepared dataset.
- The immediate next step is to expand from `1` record to a small multi-record smoke to look for record-dependent failures specific to the `lgbo` sampler path.

### LGBO Multi-record Evaluation

- Ran a small multi-record `lgbo` evaluation on the first `5` records of `hotpotqa_100`.
- Setup:
  - build indexes once
  - run `1 trial` per record
  - keep `sampler='lgbo'`
  - keep `Qwen/Qwen3-8B` as the only configured LLM

### LGBO Multi-record Results

- Aggregate over successful records:
  - `count_ok = 3`
  - `count_error = 2`
  - `mean_accuracy = 33.33`
  - `mean_f1 = 45.45`
  - `mean_precision = 42.86`
  - `mean_recall = 50.0`
  - `mean_em = 33.33`
- Per-record outcomes:
  - `id=0`: failed with reranker GPU OOM (`echorank`)
  - `id=1`: success
    - `accuracy=0.0`
    - `f1=36.36`
    - `precision=28.57`
    - `recall=50.0`
    - `em=0.0`
  - `id=2`: failed with reranker GPU OOM (`echorank`)
  - `id=3`: success
    - all short-form metrics `0.0`
  - `id=4`: success
    - all short-form metrics `100.0`

### Current Practical Interpretation

- The `lgbo` sampler itself remained runnable across the batch.
- The dominant failure mode in this run was no longer LGBO logic, but reranker initialization on GPU:
  - especially `echorank` (`flan-t5-large`) causing `torch.OutOfMemoryError`
- Some rerankers also degraded with CUDA OOM during reranking but the pipeline continued via fallback and still produced metrics.
- So the current bottleneck for broader LGBO evaluation is now the reranker search space / device behavior, not the initial LGBO Optuna integration.

### LGBO Evaluation With Restricted Rerankers

- Tried the first mitigation strategy: keep only a small reranker subset during evaluation.
- For this run, I restricted the runtime reranker search space to:
  - `flashrank`
  - `upr`
- I did this at runtime for the evaluation script, without changing public interfaces.

### Restricted-reranker LGBO Results

- Re-ran the first `5` records with:
  - `sampler='lgbo'`
  - `1 trial` per record
  - reranker pool limited to `['flashrank', 'upr']`
- Aggregate result:
  - `count_ok = 5`
  - `count_error = 0`
  - `mean_accuracy = 20.0`
  - `mean_f1 = 31.19`
  - `mean_precision = 27.99`
  - `mean_recall = 44.0`
  - `mean_em = 20.0`
- Per-record metrics:
  - `id=0`: `accuracy=0.0`, `f1=6.25`, `precision=3.70`, `recall=20.0`, `em=0.0`
  - `id=1`: `accuracy=0.0`, `f1=36.36`, `precision=28.57`, `recall=50.0`, `em=0.0`
  - `id=2`: `accuracy=100.0`, `f1=100.0`, `precision=100.0`, `recall=100.0`, `em=100.0`
  - `id=3`: all short-form metrics `0.0`
  - `id=4`: `accuracy=0.0`, `f1=13.33`, `precision=7.69`, `recall=50.0`, `em=0.0`

### Interpretation After Restriction

- Restricting the reranker pool removed the previous hard failures from heavy reranker initialization.
- `flashrank` remained the most stable option in this run.
- `upr` still triggered CUDA OOM during reranking on some records, but those failures were soft failures:
  - reranking logged an error
  - the pipeline fell back and still produced final metrics
- So this mitigation clearly improved evaluation stability, even though answer quality is still mixed.

### Single-record 10-trial LGBO Trace Check

- Ran a focused verification on a single record (`id=0`) for `10` sequential trials using:
  - `sampler='lgbo'`
  - reranker restricted to `flashrank`
  - one persistent tuner / study instance
- Added temporary runtime instrumentation to log the prompt context passed into `build_lgbo_numeric_prompt(...)` on every trial:
  - number of `history_lines`
  - whether `previous_reasoning` was present

### What This Verified

- LGBO-specific trace metadata is being written to Optuna trial user attrs.
- But completed numeric trial history is not being fed back into later LGBO prompts.

### Evidence

- Prompt instrumentation output:
  - trial `1`: `history_items = 0`, `has_previous_reasoning = false`
  - trials `2..10`: `history_items = 0`, `has_previous_reasoning = true`
- This means:
  - prior reasoning is successfully persisted and reused
  - but prior numeric observations are still not being surfaced as LGBO history lines

### Optuna Storage Observation

- After the 10-trial run, completed trials in the Optuna study had:
  - populated `suggested:*` user attrs
  - populated `lgbo_preference_raw`
  - populated `lgbo_plan`
  - populated `lgbo_reasoning`
- However, all completed trials still had:
  - `params_keys = []`
- This is the critical gap.

### Current Root Cause

- `LGBOSampler` history extraction reads numeric observations from `trial.params`.
- But the current custom-sampler integration path manually computes params and only stores them in trial user attrs (`suggested:*` / `flow`), not in Optuna's canonical `trial.params`.
- Therefore:
  - Optuna does store LGBO trace metadata
  - but the current LGBO history adapter sees no numeric parameter history
  - so `history_lines` stay empty forever

### Practical Conclusion

- The current implementation is only partially using the intended LGBO mechanism:
  - yes: LLM prompting, reasoning carry-over, plan persistence, Optuna user-attr trace storage
  - no: cross-trial numeric observation feedback inside the same study
- The next real fix should make completed sampled params readable by later LGBO trials, either by:
  - persisting sampled params in a form the history adapter reads, or
  - teaching the history adapter to reconstruct numeric params from stored trial attrs / flow metadata

### LGBO History Extraction Unified With LLMBO

- Updated `Tuner/BOTuner/lgbo_components/history.py` to follow the same history-recovery idea used in `LLMBO.py`.
- Instead of relying only on `trial.params`, the LGBO history adapter now:
  - first checks `trial.params`
  - then falls back to parsing `trial.user_attrs["flow"]`
  - then falls back to `suggested:*` user attrs if needed
- This keeps LGBO aligned with the current custom-sampler integration pattern already used by the repository.

### Post-fix Verification

- Re-ran a focused single-record LGBO experiment with:
  - `4` sequential trials
  - reranker fixed to `flashrank`
  - temporary runtime instrumentation to print prompt history usage
- Result:
  - trial 1 prompt: `history_items = 0`
  - trial 2 prompt: `history_items = 1`
  - trial 3 prompt: `history_items = 2`
  - trial 4 prompt: `history_items = 3`
- This confirms that later LGBO trials are now reading completed earlier trial trajectories within the same study.

### Optuna Trace Status After Fix

- Completed trials still store LGBO trace metadata in Optuna user attrs:
  - `lgbo_preference_raw`
  - `lgbo_plan`
  - `lgbo_reasoning`
  - `flow`
- With the adapter fix, those stored traces are now sufficient for LGBO to reconstruct numeric trial history and use it in subsequent prompts.

### Single-record 10-trial LGBO Run

- Ran a formal single-record LGBO evaluation after the history-extraction fix with:
  - record `id=0`
  - `10` sequential trials
  - reranker fixed to `flashrank`
  - HF CPU embeddings
- Result:
  - `count_ok = 10`
  - `count_error = 0`
  - `mean_accuracy = 0.0`
  - `mean_f1 = 6.05`
  - `mean_precision = 4.22`
  - `mean_recall = 12.0`
  - `mean_em = 0.0`
  - `best_f1 = 20.0`
  - `best_accuracy = 0.0`

### Interpretation Of 10-trial Run

- The post-fix LGBO loop is now operational across multiple sequential trials within the same study.
- However, on this specific record the quality signal remains weak:
  - no trial achieved non-zero `accuracy`
  - some trials improved token overlap (`f1`) up to `20.0`
  - but predictions still did not contain the gold answer string required by the `accuracy` metric
- So the main blocker has moved from trajectory wiring to answer quality / retrieval-generation effectiveness on this question.

### LGBO Ten-record Evaluation With FlashRank

- Ran a broader horizontal evaluation on the first `10` records with:
  - `sampler='lgbo'`
  - `1 trial` per record
  - reranker fixed to `flashrank`
  - HF CPU embeddings
- Result:
  - `count_ok = 10`
  - `count_error = 0`
  - `mean_accuracy = 50.0`
  - `mean_f1 = 54.86`
  - `mean_precision = 56.65`
  - `mean_recall = 65.0`
  - `mean_em = 40.0`
  - `best_accuracy = 100.0`
  - `best_f1 = 100.0`

### Per-record Snapshot

- `id=0`: all short-form metrics `0.0`
- `id=1`: `accuracy=0.0`, `f1=26.67`
- `id=2`: `accuracy=0.0`, `f1=14.81`
- `id=3`: all short-form metrics `0.0`
- `id=4`: all short-form metrics `100.0`
- `id=5`: `accuracy=100.0`, `f1=57.14`, `em=0.0`
- `id=6`: all short-form metrics `100.0`
- `id=7`: all short-form metrics `100.0`
- `id=8`: all short-form metrics `100.0`
- `id=9`: `accuracy=0.0`, `f1=50.0`

### Interpretation

- Once reranker instability was removed by fixing the reranker pool to `flashrank`, the LGBO path became stable across a wider slice of the dataset.
- The quality is still uneven across records, but the overall picture is much better than the earlier noisy runs that mixed in heavy GPU-sensitive rerankers.

### Current LGBO vs Reference `lgbo/` And Paper

- Re-reviewed the current HEART-side `LGBOSampler` against the reference `lgbo/` implementation:
  - `lgbo/prompt.py`
  - `lgbo/exp.py`
  - `lgbo/decide.py`
  - `lgbo/boo.py`
  - `lgbo/prior.py`

### What Is Already Aligned

- HEART now does implement the main paper-style outer loop skeleton:
  - collect history
  - build an LGBO prompt
  - call an LLM
  - parse a `point` / `region` style preference
  - convert that preference into an internal plan
  - generate the next candidate
  - persist reasoning / raw preference / plan into Optuna user attrs
- HEART now also reuses within-study history across sequential LGBO trials:
  - previous reasoning is reused
  - numeric completed-trial history is now reconstructed from Optuna storage

### Main Remaining Gaps To `lgbo/` Code

- HEART still does **not** implement the BoTorch-based posterior tilting path used by the reference code:
  - no `LinearExponentialRegionalMeanTiltPlugAndPlay`
  - no `TiltedModel`
  - no region mean-tilt posterior update
  - no confidence-to-`delta` calibration in the paper/reference sense
- HEART candidate generation is still a lightweight heuristic adapter, not the reference acquisition logic in `lgbo/boo.py`:
  - current HEART path uses midpoint / interpolation / best-observation fallback
  - reference `lgbo/boo.py` uses tilted posterior + EI/TS-style BO proposal machinery
- HEART planner is a simplified approximation of `lgbo/decide.py`:
  - current point guidance becomes a compact `region-soft`
  - but it does not reproduce the reference code's explicit `point` / `region-soft` / `region` switching based on expected grid coverage and confidence-to-delta scaling
- HEART prompting is intentionally much simpler than the reference prompt stack:
  - current prompt is numeric-only and flat
  - reference `lgbo/prompt.py` carries richer domain instructions, stricter formatting guards, and iterative review/update prompt construction
- HEART currently only applies LGBO to the numeric subspace:
  - categorical / structural parameters still come from generic search-space sampling
  - reference LGBO logic assumes the expert recommendation directly shapes the BO search proposal itself

### Main Remaining Gaps To The Paper-Level Method

- No full posterior reweighting / region-lift mechanism is currently active in HEART.
- No BoTorch-based acquisition over a tilted surrogate is currently active in HEART.
- No paper-level batch proposal logic (`q > 1`) or pathwise posterior sampling is currently ported.
- No explicit baseline-vs-LGBO comparison loop like the reference toy experiments in `lgbo/exp.py`.
- No paper-style mechanism for richer iterative review prompts that explicitly carry forward historical observations, reasoning summaries, and action-specific next-round focus beyond the current lightweight prompt builder.

### Practical Bottom Line

- The current HEART implementation is now best described as:
  - **LGBO-style prompt-guided optimization control**
  - built on HEART's existing Optuna runtime
  - with Optuna-backed trace persistence and within-study history reuse
- It is **not yet** the full paper/reference LGBO algorithm, because the core surrogate/acquisition side from `lgbo/prior.py` + `lgbo/boo.py` has not been ported.

### LGBO Surrogate / Acquisition Integration

- Implemented the next integration step requested after the gap review:
  - keep the current numeric-only LGBO scope
  - add a real surrogate + acquisition path under the existing sampler interface

#### Dependency update

- Installed BO dependencies into the active Python environment:
  - `botorch`
  - `gpytorch`

#### Code changes

- Added `Tuner/BOTuner/lgbo_components/surrogate.py`
  - ports a minimal reusable subset of the reference `lgbo/prior.py` / `lgbo/boo.py` path into HEART
  - includes:
    - `SingleTaskGP` surrogate fitting
    - standardized objective handling
    - region-lift posterior tilt via `LinearExponentialRegionalMeanTiltPlugAndPlay`
    - `TiltedModel` wrapper
    - `qLogExpectedImprovement` acquisition over a Sobol candidate pool
    - confidence -> `delta` conversion for tilt strength
- Updated `Tuner/BOTuner/lgbo_components/candidate.py`
  - added an optional `use_bayesian_surrogate` path
  - candidate generation now tries:
    - surrogate fitting on completed numeric trials
    - acquisition optimization with the parsed LGBO plan
    - heuristic fallback only if surrogate/acquisition cannot run
  - records which path was used in `last_strategy`
- Updated `Tuner/BOTuner/LGBO.py`
  - enabled the surrogate/acquisition path for normal LGBO sampling
  - stores `candidate_strategy` in the saved LGBO trial plan trace so later debugging can tell whether the sampler used:
    - `bayes_surrogate`
    - or heuristic fallback

#### Verification

- `python tests/test_lgbo_components.py`
  - passed
- `python -m py_compile Tuner/BOTuner/LGBO.py Tuner/BOTuner/lgbo_components/candidate.py Tuner/BOTuner/lgbo_components/surrogate.py tests/test_lgbo_components.py`
  - passed
- Ran an explicit smoke script against `LGBOCandidateGenerator(..., use_bayesian_surrogate=True)`
  - first hit a `botorch` compatibility issue because acquisition called `posterior()` with extra kwargs
  - fixed the tilted-posterior wrapper to accept and forward those kwargs
  - reran successfully
  - confirmed `last_strategy = {'mode': 'bayes_surrogate', 'plan_mode': 'region-soft'}`

#### Current state after this pass

- HEART LGBO is still numeric-only, but it no longer stops at prompt parsing + heuristic midpoint selection.
- For numeric completed-trial history, HEART can now:
  - fit a GP surrogate
  - apply a region-lift style posterior tilt from the LGBO plan
  - choose the next candidate via acquisition instead of heuristic-only selection
- This is still a narrowed integration of the reference method, not yet a full port of every prior/sampler variant in `lgbo/prior_monte_carlo.py`.

### Real LGBO Smoke After Surrogate Integration

- Confirmed again that the example dataset is present and valid:
  - `dataset/hotpotqa_100_records.json`
  - verified record count: `100`
- Ran a real HEART-side `lgbo` smoke on:
  - dataset: `hotpotqa_100`
  - records: first `1`
  - trials per record: `1`
  - reranker fixed to `flashrank`
  - embedding switched to HF CPU:
    - `sentence-transformers/all-MiniLM-L6-v2`
    - dimensions `384`
  - HF cache redirected into the repo workspace to avoid permission issues

#### Runtime issues resolved during this smoke

- The first attempt failed because `Common.Logger` still initialized from the default `/mnt/data/...` path at import time.
  - worked around this in the smoke script by redirecting the bootstrap default config to the workspace before importing runtime modules
- The next attempt failed because HF embedding cache writes were going to `~/.cache/huggingface/...` and hit permissions.
  - redirected cache into `HEART/cache/huggingface`
- Another attempt failed because FAISS config validation required an explicit integer embedding dimension.
  - fixed runtime config for this smoke to use `384`
- One attempt also failed because the temporary smoke script had not pre-created `Results/`.
  - fixed by creating the expected run directories before evaluation

#### Final smoke result

- The real `lgbo` path completed end-to-end with exit code `0`.
- Observed stages succeeded:
  - corpus chunking
  - FAISS index load/build
  - BM25 index load/build
  - Optuna study creation
  - `lgbo` trial sampling
  - retrieval + `flashrank` reranking
  - response generation
  - short-form evaluation
- Final metrics on record `id=0`:
  - `accuracy = 0.0`
  - `f1 = 11.7647`
  - `precision = 8.3333`
  - `recall = 20.0`
  - `em = 0.0`

#### Important interpretation

- The trial attrs confirm that `lgbo` metadata is being written during the real HEART run:
  - `lgbo_plan`
  - `lgbo_preference_raw`
  - `lgbo_reasoning`
- However, this `1 query x 1 trial` smoke did **not** use the new surrogate path yet:
  - `lgbo_plan.candidate_strategy.mode = "heuristic"`
  - `lgbo_plan.mode = "numeric_v1_fallback"`
- The immediate cause in this run was prompt parsing failure from the LLM output:
  - stored error: `malformed node or string on line 1: <ast.Name object ...>`
- Also, with only one trial there is no completed within-study numeric history yet, so even a clean second-step surrogate proposal would need at least one more completed trial on the same record to be meaningful.

#### Current conclusion

- The example dataset is confirmed present (`100` records).
- The real HEART `lgbo` runtime still works after the surrogate/acquisition integration.
- The next focused verification should be:
  - run the same record for multiple sequential trials
  - inspect whether `candidate_strategy.mode` flips from heuristic to `bayes_surrogate`
  - and separately harden LGBO preference parsing so first-trial LLM outputs do not fall back unnecessarily

### Same-query Sequential Trial Check

- Ran the exact follow-up verification requested by the user:
  - fixed the same record (`id=0`)
  - ran `3` sequential `lgbo` trials in one persistent study/tuner
  - kept reranker fixed to `flashrank`
  - kept the same HF CPU embedding setup used in the previous smoke
- Goal:
  - verify whether later trials can use prior completed trials and switch from heuristic selection to the new surrogate/acquisition path

#### Sequential trial results

- Trial 1:
  - `candidate_strategy.mode = "heuristic"`
  - `lgbo_plan.mode = "numeric_v1_fallback"`
  - metrics:
    - `accuracy = 0.0`
    - `f1 = 11.7647`
- Trial 2:
  - `candidate_strategy.mode = "heuristic_fallback"`
  - explicit fallback reason:
    - `Need at least two completed observations for surrogate fitting`
  - `lgbo_plan.mode = "numeric_v1_fallback"`
  - metrics:
    - `accuracy = 0.0`
    - `f1 = 6.6667`
- Trial 3:
  - `candidate_strategy.mode = "bayes_surrogate"`
  - `plan_mode = "numeric_v1_fallback"`
  - sampled numeric candidate changed to:
    - `rag_top_k = 126`
    - `rag_hybrid_bm25_weight = 0.1`
    - `rag_query_decomposition_num_queries = 2`
    - `reranker_top_k = 123`
  - metrics:
    - `accuracy = 0.0`
    - `f1 = 0.0`

#### Interpretation

- This confirms the intended cross-trial behavior is now present:
  - later trials on the same query do read prior completed observations
  - the surrogate/acquisition path is not dead code
  - after enough completed history exists, `LGBOCandidateGenerator` can switch into `bayes_surrogate`
- The threshold behavior observed in practice is:
  - first trial: no prior history, so heuristic only
  - second trial: only one completed observation available, still not enough for surrogate fitting
  - third trial: two completed observations available, surrogate fitting becomes possible and is used

#### Remaining issue exposed by this check

- Even when the surrogate path activates, the LLM preference parser is still frequently failing on the raw `Final Answer` text:
  - repeated stored error:
    - `malformed node or string on line 1: <ast.Name object ...>`
- So the current sampler behavior is:
  - history-aware surrogate proposal works
  - but the prompt/planning side often collapses into `numeric_v1_fallback`
  - meaning the surrogate is currently being driven mostly by fallback-mode history rather than clean parsed point/region guidance

### Cross-query History Reuse

- Implemented the requested cross-query LGBO history sharing behavior.
- Before this change:
  - each query created its own Optuna study using `study_name = query['id']`
  - so later queries could not see earlier queries' completed trials
- After this change:
  - `lgbo` now uses one shared experiment-level study:
    - `{exp_name}__lgbo_shared_history`
  - this shared study is reused across queries inside the same experiment directory
  - the latest current query is still written into study user attrs for prompt construction
- Important scope choice:
  - cross-query history is now shared for surrogate/candidate generation
  - previous free-form reasoning is filtered to the current query only, to avoid blindly carrying natural-language rationale from an unrelated earlier question

### Parser Hardening

- Fixed `Tuner/BOTuner/lgbo_components/preference.py` so LGBO no longer depends on quoted `"point"` / `"region"` literals.
- The parser now supports:
  - bareword forms like:
    - `[point, [65, 0.5, 10, 65], 1.0]`
    - `[region, [[...], [...]], 0.5]`
  - transposed region-pair payloads like:
    - `[[low1, high1], [low2, high2], ...]`
    - which are converted into the internal `[lb_list, ub_list]` form
- Also updated history rendering to tag entries with `query_id=...` in prompt history lines for clearer cross-query provenance.

### Validation After Cross-query + Parser Fixes

- Ran targeted component verification:
  - `python tests/test_lgbo_components.py`
  - result: passed (`10` tests)
- Added test coverage for:
  - query-scoped reasoning lookup
  - bareword `point` parsing
  - transposed `region` payload parsing

### Real Cross-query LGBO Verification

- Ran a real HEART-side validation over the first `3` records of `hotpotqa_100`:
  - `1` trial per query
  - `sampler='lgbo'`
  - reranker fixed to `flashrank`
  - HF CPU embeddings with local writable cache
  - one shared Optuna study across all three queries

#### Observed trial-by-trial behavior

- Query 0:
  - `candidate_strategy.mode = "heuristic"`
  - parser succeeded
  - `lgbo_plan.mode = "region"`
- Query 1:
  - `candidate_strategy.mode = "heuristic_fallback"`
  - explicit reason:
    - `Need at least two completed observations for surrogate fitting`
  - this confirms query 1 did see query 0 history, but one prior completed point is still insufficient for the GP surrogate
  - parser succeeded
  - `lgbo_plan.mode = "region"`
- Query 2:
  - `candidate_strategy.mode = "bayes_surrogate"`
  - `plan_mode = "region"`
  - sampled numeric candidate changed substantially:
    - `rag_top_k = 127`
    - `rag_hybrid_bm25_weight = 0.8`
    - `rag_query_decomposition_num_queries = 16`
    - `reranker_top_k = 2`
  - this confirms query 2 used the first two queries' completed trial history for surrogate-based proposal

#### Final interpretation

- The requested cross-query behavior is now present:
  - query 2 can use query 1 history
  - query 3 can use query 1 + query 2 history
  - and the shared-history mechanism will continue in the same pattern for later queries in the same experiment
- The parser issue that previously forced many trials into `numeric_v1_fallback` has been substantially improved:
  - in this 3-query validation, all three queries produced parseable LGBO plans
  - and the third query successfully reached `bayes_surrogate` with `plan_mode = "region"`

### Attempted 5-query x 5-trial Run

- Tried to run a larger verification requested by the user:
  - first `5` queries
  - `5` trials per query
  - shared LGBO study across all queries
  - `flashrank` reranker only
  - HF CPU embeddings with local cache
- Attempted this twice:
  - `lgbo_cross_query_5records_5trials`
  - `lgbo_cross_query_5records_5trials_retry1`

#### Outcome

- Both runs failed before finishing the first query.
- The failure was not caused by the new cross-query LGBO logic.
- The blocker was repeated outbound LLM API connectivity failure during generation:
  - `openai.APIConnectionError: Connection error.`
  - underlying cause:
    - `httpx.ConnectError: [Errno -3] Temporary failure in name resolution`
    - target host:
      - `https://api.siliconflow.cn/v1/chat/completions`
- The failures happened after indexing and study creation had already succeeded, so the run reached:
  - chunking
  - FAISS/BM25 indexing
  - shared-study creation
  - trial setup
  - retrieval / reranking
  - but failed during response synthesis LLM calls

#### Practical interpretation

- The previously completed `3-query x 1-trial` run remains the latest successful end-to-end confirmation that:
  - cross-query shared history works
  - query 2 can consume query 1 history
  - query 3 can consume query 1 + query 2 history
  - and query 3 can switch into `bayes_surrogate`
- The requested larger `5 x 5` validation is currently blocked by external API/network instability rather than by an identified new code regression.

### SiliconFlow Connectivity Debugging

- Investigated the repeated `openai.APIConnectionError` seen during larger LGBO runs.

#### Root cause isolation

- In the default sandboxed shell environment:
  - DNS resolution failed not only for `api.siliconflow.cn`, but also for common domains like:
    - `google.com`
    - `pypi.org`
    - `github.com`
  - proxy environment variables were set to a dead local proxy endpoint:
    - `HTTP_PROXY=http://127.0.0.1:43485`
    - `HTTPS_PROXY=http://127.0.0.1:43485`
    - `ALL_PROXY=http://127.0.0.1:43485`
  - the local proxy port itself was closed (`Connection refused`)
- Even after stripping proxy variables inside the sandbox, DNS still failed there, so the sandboxed runtime itself was not usable for stable outbound API access.

#### Verification outside sandbox

- Re-tested the same connectivity outside the sandbox:
  - `google.com`, `pypi.org`, `github.com`, and `api.siliconflow.cn` all resolved successfully
  - `/etc/resolv.conf` was present and pointed to the local systemd resolver
  - `curl https://api.siliconflow.cn/v1/models` reached the server successfully
  - a real authenticated chat completion request to SiliconFlow succeeded and returned:
    - `OK`

#### Practical conclusion

- SiliconFlow itself is reachable and working.
- The blocker was environmental:
  - broken DNS / proxy behavior in the sandboxed shell context
- For future real LGBO runs that need outbound LLM access, the stable workaround is:
  - run them outside the sandbox
  - and avoid the broken local proxy setup used by the sandboxed environment

### Completed 5-query x 5-trial Non-sandbox Validation

- Ran the previously blocked larger validation outside the sandbox:
  - experiment: `lgbo_cross_query_5records_5trials_nonsandbox`
  - dataset slice: first `5` records of `hotpotqa_100`
  - trials: `5` per query
  - shared Optuna study across all queries
  - reranker pool restricted to `flashrank`
  - response LLM pool restricted to `Qwen/Qwen3-8B`
  - HF CPU embeddings with local writable cache
- The run finished successfully:
  - exit code `0`
  - total completed trials: `25 / 25`

#### Trial-by-trial history reuse evidence

- Query 0 / Trial 1:
  - no prior observations exist yet
  - `candidate_strategy.mode = "heuristic"`
- Query 0 / Trial 2:
  - one completed prior trial is already visible
  - the run explicitly reported:
    - `candidate_strategy.mode = "heuristic_fallback"`
    - `error = "Need at least two completed observations for surrogate fitting"`
  - this confirms trial 2 did use prior history, but one point was still insufficient for GP fitting
- Query 0 / Trial 3:
  - now there are two completed prior observations
  - `candidate_strategy.mode = "bayes_surrogate"`
- Query 1 / Trial 1:
  - `study_trials_total = 6`
  - `candidate_strategy.mode = "bayes_surrogate"`
  - this confirms the first trial of query 1 already consumed all 5 trials from query 0
- Query 2 / Trial 1:
  - `study_trials_total = 11`
  - `candidate_strategy.mode = "bayes_surrogate"`
  - this confirms query 2 consumed the full cumulative history from queries 0 and 1
- Query 3 / Trial 1:
  - `study_trials_total = 16`
  - `candidate_strategy.mode = "bayes_surrogate"`
- Query 4 / Trial 1:
  - `study_trials_total = 21`
  - `candidate_strategy.mode = "bayes_surrogate"`
- Final trial:
  - Query 4 / Trial 5 ended with `study_trials_total = 25`
  - `candidate_strategy.mode = "bayes_surrogate"`

#### Practical conclusion from the 5x5 run

- Every trial after the very first one had access to prior completed results through the shared Optuna study.
- The only trial that still could not fit the surrogate was Query 0 / Trial 2, because only one completed observation existed at that point.
- From Query 0 / Trial 3 onward, the Bayesian surrogate was active.
- Therefore:
  - `1 / 25` trial had no history available
  - `1 / 25` trial used history but had to fall back because there was only one prior point
  - `23 / 25` trials used `bayes_surrogate`
- This is the strongest end-to-end confirmation so far that the requested cross-query history sharing is working in real HEART runs.

### Current LGBO Optimization Surface

- Important distinction:
  - the full Optuna trial contains both categorical and numeric knobs
  - the current LGBO implementation is still `numeric-only` on the surrogate side
  - categorical fields can still vary in the final trial config, but they are not modeled by the current GP/acquisition loop

#### Parameters currently modeled by the LGBO surrogate

- The current surrogate only consumes distributions that are `IntDistribution` or `FloatDistribution`.
- In the current HEART search space and current `5x5` run, those numeric parameters are:
  - `rag_top_k`
    - range: `2 .. 128`
    - step: `1`
  - `rag_hybrid_bm25_weight`
    - range: `0.1 .. 0.9`
    - step: `0.1`
  - `rag_query_decomposition_num_queries`
    - range: `2 .. 20`
    - step: `2`
  - `reranker_top_k`
    - range: `2 .. 128`
    - step: `1`
- These are the parameters that are:
  - included in `numeric_specs`
  - summarized into LGBO history lines
  - passed into the LLM preference parser/planner
  - modeled by the GP surrogate
  - proposed by the acquisition-driven candidate generator

#### Parameters present in the overall trial but not modeled by the current surrogate

- The following trial fields can still vary in the final Optuna config, but are not part of the current numeric LGBO surrogate:
  - `template_name`
  - `response_synthesizer_llm`
  - `rag_method`
  - `rag_query_decomposition_enabled`
  - `rag_query_decomposition_llm_name`
  - `rag_fusion_mode`
  - `reranker_name`

#### Effective parameter freedom in the non-sandbox 5x5 run

- Because the run was intentionally restricted for stability, some categorical dimensions were effectively fixed to one choice:
  - `response_synthesizer_llm = Qwen/Qwen3-8B`
  - `rag_query_decomposition_llm_name = Qwen/Qwen3-8B`
  - `reranker_name = flashrank`
- The parameters that still visibly changed across completed trials in this run were:
  - categorical / boolean:
    - `template_name`
    - `rag_method`
    - `rag_query_decomposition_enabled`
    - `rag_fusion_mode`
  - numeric surrogate-controlled:
    - `rag_top_k`
    - `rag_hybrid_bm25_weight`
    - `rag_query_decomposition_num_queries`
    - `reranker_top_k`

#### Current interpretation

- So the cleanest summary is:
  - the trial-level search space is mixed categorical + numeric
  - but the current LGBO surrogate/acquisition path is only optimizing the four numeric knobs above
  - categorical behavior is still outside the current surrogate model
- This matches the intended scope of the current implementation phase:
  - keep LGBO numerical for now
  - get real shared-history surrogate behavior working first
