# LGBO Integration Plan

## Goal

Close the most important gaps between `lgbo/` and `HEART/Tuner/BOTuner/LGBO.py` without changing the existing public sampler interface:

- keep `LGBOSampler.__init__`
- keep `infer_relative_search_space(...)`
- keep `sample_relative(...)`
- keep the current `OptunaTuner` integration path

## Current Gaps

### 1. Prompt loop is not connected

Current `LGBOSampler.sample_relative(...)` never builds an LGBO prompt and never calls an LLM.

Impact:
- no continuous guidance
- no query-aware behavior
- no use of trial history beyond a fallback candidate

### 2. Parser only handles raw bracket literals

Current `LGBOPreferenceParser` parses only exact Python-list style inputs and cannot safely extract:

- `Thinking: ...`
- `Final Answer: ...`
- mixed natural language + bracket output

Impact:
- brittle against realistic LLM responses
- cannot persist reasoning traces

### 3. Planner is too shallow

Current planner only maps:

- point -> point
- region -> region

But the reference `lgbo/decide.py` does more:

- converts point preferences into a compact soft region / local preference shape
- clips and regularizes region bounds
- uses confidence as preference strength

Impact:
- current implementation does not preserve the paper-inspired “point as local region guidance” idea

### 4. Candidate generation ignores objective structure

Current candidate generation:

- uses point directly
- uses region midpoint
- otherwise returns the last observation or domain midpoint

Missing:
- objective-aware fallback using the best completed trial
- confidence-aware interpolation between exploration and exploitation
- region-soft behavior

### 5. Trace store is underused

Current trace writing stores only a synthetic fallback plan.

Missing:
- raw model output
- parsed preference
- reasoning trace
- actual plan chosen by the sampler

## Planned Implementation

### Phase 1. Connect numeric prompt guidance

- enrich `Prompt/LGBOPrompt.py` with a reusable numeric LGBO system prompt
- generate prompt inputs from:
  - Optuna query metadata
  - numeric parameter specs
  - recent completed trial history
  - latest stored reasoning trace

### Phase 2. Harden parsing

- extend `LGBOPreferenceParser` to extract:
  - `Thinking`
  - `Final Answer`
  - a raw bracket preference from noisy text
- keep support for the existing exact literal format

### Phase 3. Improve internal planning

- convert point preferences into a compact local box (`region-soft`) while preserving point semantics
- keep region clipping and ordering stable within parameter bounds
- propagate confidence into the internal plan

### Phase 4. Improve lightweight candidate generation

- use the best completed numeric observation as the default fallback
- support:
  - `point`
  - `region`
  - `region-soft`
- keep the candidate generator dependency-light and avoid changing sampler interfaces

### Phase 5. Wire into `LGBOSampler`

- lazily create an LLM provider from `config.llms[0]`
- call the LLM from inside `sample_relative(...)`
- gracefully fall back to numeric-only behavior if:
  - no LLM is configured
  - the call fails
  - parsing fails

### Phase 6. Verification and logging

- add unit coverage for:
  - prompt parsing from realistic text
  - point -> region-soft planning
  - objective-aware fallback candidate generation
- append implementation details and verification results to `agent_workspace/Progress.md`

## Non-Goals For This Pass

- do not replace the lightweight candidate generator with the full BoTorch tilt posterior from `lgbo/prior.py`
- do not change `OptunaTuner` public behavior
- do not redesign the existing search-space interface
