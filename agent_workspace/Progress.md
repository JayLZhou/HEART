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
