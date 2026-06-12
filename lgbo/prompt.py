# prompt.py
"""
Prompt definitions for chemistry experiment optimization assistant.
"""
import re
from typing import Callable, Any, Iterable, Sequence

SYSTEM_PROMPT = """You are a scientist specializing in experimental optimization.
Your job: read the experiment background & the latest review, think carefully, and recommend the next experiment.

# Evidence hierarchy (critical)
- PRIMARY: Background knowledge, physical/chemical mechanisms, constraints, and units.
- SECONDARY (auxiliary only): Historical trial points/observations and thinking in the review.
- If background implications conflict with historical points, SIDE WITH BACKGROUND.

# Background (fixed across rounds)
- We run iterative chemistry experiments (e.g., polymerization/hydrolysis/organic synthesis).
- Parameters are physical and NOT normalized. Always use the declared units & order.

# Modes (pick exactly ONE)
1) [point, [x1, x2, ..., xd], confidence]
2) [region, [[lb1, lb2, ..., lbd],
             [ub1, ub2, ..., ubd]], confidence]
- confidence ∈ [0,1].
- In region mode, each dimension must have (lb ≤ ub) and follow the declared parameter order & units.
- For categorical variables, output their literal value (e.g., "DMF") in both point and region.
  If a category is fixed in region, set lb=ub to that same literal value.

# How to reason (prioritize background over past points)
- Start from first principles: mechanism-driven trends, feasible/unsafe ranges, known monotonicities, interactions.
- Use historical data ONLY as weak corroboration or disproof of a background-based hypothesis.
- Do NOT anchor on previous best/nearest points; avoid proposing a point merely because it appeared before.
- If historical points cluster narrowly, consider a background-justified exploratory move (e.g., shift in a mechanism-relevant factor).
- Prefer REGION when background suggests multiple nearby settings could satisfy the mechanistic target; choose POINT only when background+data imply a sharp optimum.

# Output protocol (two blocks)
1) Thinking:
   - Be concise but informative, in this order:
     (a) Background-based rationale (mechanism/constraints) that leads to your proposal.
     (b) How (if at all) historical data supports/contradicts this mechanism (≤2 sentences).
     (c) Why point vs region given the mechanism and uncertainty.
2) Final Answer:
   - Strict structure with no extra words:
     [point, [x1, x2, ..., xd], ccc]
     OR
     [region, [[lb1, lb2, ..., lbd],
               [ub1, ub2, ..., ubd]], ccc]

# Hard constraints
- Do NOT normalize or re-order parameters.
- Keep units consistent with the declared parameter order.
- No extra commentary in Final Answer beyond the bracketed structure.

# Anti-collapse checks
- Never center a region or point on a past observation unless mechanistically justified.
- If you reuse a past setting, explicitly state the mechanism that makes it optimal (in Thinking).
"""

TOY_SYSTEM_PROMPT = """You are an optimization scientist assisting with classic black-box toy functions (Rastrigin, Ackley, Griewank, Lévy, etc.).
Your job: read the background & review, think carefully, and recommend the next evaluation that MINIMIZES f(x).

# Evidence hierarchy (critical)
- PRIMARY: Known mathematical structure of the target function class (periodicity, separability, conditioning, multimodality).
- - SECONDARY (auxiliary only): Historical trial points/observations and thinking in the review.
- If heuristic from function structure conflicts with sparse/noisy history, SIDE WITH FUNCTION STRUCTURE.

# Problem setting (toy BO)
- Single objective: MINIMIZE f(x). Smaller f(x) is strictly better. Never maximize, never flip signs.
- Continuous variables only, real-valued and dimensionless.
- Parameter order is [x1, x2, ..., xd]. Use this order exactly. Do NOT normalize or reorder.
- Respect bounds strictly; if you propose a point/region partly outside bounds, snap to the nearest feasible values before output.

# Two allowed answer modes (pick exactly ONE)
1) [point, [x1, x2, ..., xd], confidence]
2) [region, [[lb1, lb2, ..., lbd],
             [ub1, ub2, ..., ubd]], confidence]
- confidence ∈ [0,1].
- Region: axis-aligned, lb_i ≤ ub_i, all within bounds.

# How to reason (background-first)
- Derive a proposal primarily from the function class: e.g., Rastrigin (periodic basins near integer grids), Ackley (broad flat with global min near 0), Griewank (product-cosine interactions), Lévy (global min at 0).
- Use history ONLY to (i) avoid clearly poor basins or (ii) refine inside a function-justified promising basin.
- Do NOT anchor on previous best; avoid proposing points solely because they are near incumbents.
- Early: explore function-structure-informed basins lacking coverage that could LOWER f(x).
- Later: refine around basins that the function structure AND observations jointly support.

# When to choose point vs region
- point: structure indicates a sharp candidate (e.g., near 0 for Ackley/Lévy) and history does not contradict it.
- region: multiple minima/basins plausible by structure; propose a compact box expected to contain LOWER f(x).

# Output protocol (two blocks)
1) Thinking:
   - 4–8 sentences, in this order:
     (a) Function-structure rationale for where LOWER f(x) likely is.
     (b) How (briefly) the observations align or conflict (≤2 sentences).
     (c) Why point vs region and why it should LOWER f(x).
2) Final Answer:
   - Strict structure with no extra words:
     [point, [x1, x2, ..., xd], ccc]
     OR
     [region, [[lb1, lb2, ..., lbd],
               [ub1, ub2, ..., ubd]], ccc]

# Hard constraints
- This is a MINIMIZATION task. Smaller f(x) is better. Do NOT maximize or negate f.
- Keep proposals feasible and in the declared parameter order. Do NOT normalize or reorder parameters.
- In region mode use axis-aligned boxes only; keep them reasonably tight (not the whole domain).

# Anti-collapse checks
- Do not replicate prior points unless the function structure uniquely favors them; if you do, justify explicitly.
- If historical samples cluster narrowly, consider a structure-informed exploratory move across periods/axes.

# Examples (format only; values are illustrative)
- [point, [0.15, -2.0, 1.3], 0.78]
- [region, [[-0.2, 1.0, -1.5],
            [ 0.1, 1.6, -0.9]], 0.65]
"""

def make_user_prompt(
    background: str,
    parameters: list[str],
    objective: str,
    constraints: str,
    history: list[str] | None = None,
    last_reasoning: str | None = None,
    key_observations: str | None = None,
    this_round_focus: str | None = None,
) -> str:
    """
    Generate the user prompt content.

    Args:
        background: description of the experiment type & steps
        parameters: list of parameter names with units, defining order
        objective: experiment objective
        constraints: textual constraints
        history: list of strings describing past experiments, newest first
        last_reasoning: summary of last round model reasoning
        key_observations: new observations since last round
        this_round_focus: special focus of this round

    Returns:
        str: formatted user prompt
    """
    parts = []
    parts.append("[Background]")
    parts.append(f"- Experiment type & steps: {background}")
    parts.append(f"- Parameter order (d={len(parameters)}):")
    parts.append(f"  [ {', '.join(parameters)} ]")
    parts.append(f"- Objective: {objective}")
    parts.append(f"- Constraints: {constraints}")
    parts.append("")
    parts.append("[Review]")
    if history:
        parts.append("- Historical data (newest first):")
        for h in history:
            parts.append(f"  {h}")
    if last_reasoning:
        parts.append(f"- Last round reasoning summary: {last_reasoning}")
    parts.append(f"- Adoption note: Suggestion was incorporated into BO surrogate model update, actual tested points may differ.")
    if key_observations:
        parts.append(f"- Key observations since last round: {key_observations}")
    if this_round_focus:    
        parts.append(f"- This round focus: {this_round_focus}")
    return "\n".join(parts)

def make_user_prompt_toy_bo(
    func_name: str,
    d: int | None = None,
    bounds: list[tuple[float, float]] | None = None,
    *,
    noise_desc: str | None = None,
    history: list[str] | None = None,
    last_reasoning: str | None = None,
    key_observations: str | None = None,
    this_round_focus: str | None = None,
) -> str:

    name = func_name.strip().lower()

    default_d = {"rastrigin": 10, "ackley": 10, "griewank": 10, "levy": 10}

    default_bounds = {
        "rastrigin": (-5.12, 5.12),
        "ackley": (-5.0, 5.0),
        "griewank": (-600.0, 600.0),
        "levy": (-10.0, 10.0),
    }

    landscape = {
        "rastrigin": (
            "Highly multimodal with a regular grid of local minima; "
            "strong risk of getting trapped in local basins; uniform periodic landscape."
        ),
        "ackley": (
            "Central deep basin surrounded by ripples; pronounced exploration–exploitation trade-off; "
            "requires escaping broad flat regions and fine local search near the center."
        ),
        "griewank": (
            "Smooth with many shallow local minima caused by multiplicative cosine terms; "
            "global trend is gentle; convergence speed and precision dominate difficulty."
        ),
        "levy": (
            "Rugged and complex with numerous local optima; sharp cliffs and plateaus; "
            "requires careful step-size control and robust exploration."
        ),
    }


    if name not in landscape:
        raise ValueError("func_name must be one of: rastrigin, ackley, griewank, levy")


    d_eff = d if d is not None else default_d[name]
    if bounds is None:
        lo, hi = default_bounds[name]
        bounds = [(lo, hi)] * d_eff
    if len(bounds) != d_eff:
        raise ValueError("bounds length must equal d")

    param_names = [f"x{i+1}" for i in range(d_eff)]
    bounds_str = ", ".join([f"{p}∈[{lo:g},{hi:g}]" for (p, (lo, hi)) in zip(param_names, bounds)])

    objective = "Minimize f(x) (single objective). The model should propose either a point or a region."
    constraints = (
        "All parameters are real-valued and dimensionless; use the declared order and bounds exactly; do not normalize.\n"
        f"- Bounds: {bounds_str}\n"
        "- If region is chosen, provide [[lb1,...,lbd],[ub1,...,ubd]] with lb_i ≤ ub_i and within bounds.\n"
        "- Keep proposals feasible; avoid violating bounds."
    )
    if noise_desc:
        constraints += f"\n- Observations may be noisy: {noise_desc}"

    background = (
        f"This task optimizes a classic black-box test function for Bayesian optimization: {name.title()}.\n"
        f"Landscape: {landscape[name]}\n\n"
        "You will recommend the next evaluation either as a single point or as a region likely to contain improved values. "
        "The function is unknown to you beyond the summary above; treat it as a black-box with the given bounds."
    )

    parts = []
    parts.append("[Background]")
    parts.append(f"- Experiment type & purpose: {background}")
    parts.append(f"- Parameter order (d={d_eff}):")
    parts.append("  [ " + ", ".join(param_names) + " ]")
    parts.append(f"- Objective: {objective}")
    parts.append(f"- Constraints: {constraints}")
    parts.append("")
    parts.append("[Review]")
    if history:
        parts.append("- Historical data (newest first):")
        for h in history:
            parts.append(f"  {h}")
    if last_reasoning:
        parts.append(f"- Last round reasoning summary: {last_reasoning}")
    parts.append("- Adoption note: Suggestions were used as guidance; actual tested points may differ.")
    if key_observations:
        parts.append(f"- Key observations since last round: {key_observations}")
    if this_round_focus:
        parts.append(f"- This round focus: {this_round_focus}")

    return "\n".join(parts)
# ---------- Parsers (assistant output & previous user prompt) ----------

_HDR_FINAL = re.compile(
    r"(?im)^\s{0,3}(?:#+\s*)?\[?\s*final\s+answer\s*\]?\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_HDR_THINK = re.compile(
    r"(?im)^\s{0,3}(?:#+\s*)?\[?\s*thinking\s*\]?\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


_POINT_ANY_RE = re.compile(
    r"\[\s*point\s*,\s*\[(.*?)\]\s*,\s*([01](?:\.\d+)?)\s*\]",
    re.IGNORECASE | re.DOTALL,
)
_REGION_ANY_RE = re.compile(
    r"\[\s*region\s*,\s*\[\[(.*?)\]\s*,\s*\[(.*?)\]\s*\]\s*,\s*([01](?:\.\d+)?)\s*\]",
    re.IGNORECASE | re.DOTALL,
)

_THINKING_RE = re.compile(
    r"(?is)"
    r"^\s{0,3}(?:#+\s*)?\[?\s*thinking\s*\]?\s*:?\s*$"
    r"(.*?)"
    r"(?=^\s{0,3}(?:#+\s*)?\[?\s*final\s+answer\s*\]?\s*:?\s*$|\Z)", 
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)

_FALLBACK_THINKING_RE = re.compile(
    r"(?is)(?:^|\n)\s*(?:#\s*)?\[?\s*thinking\s*\]?\s*:?\s*(.*?)\s*"
    r"(?=\n\s*(?:#\s*)?\[?\s*final\s*answer\s*\]?\s*:?\b|$)"
)

_PARAM_ORDER_RE = re.compile(
    r"Parameter order\s*\(d\s*=\s*\d+\)\s*:\s*\[\s*(.*?)\s*\]",
    re.IGNORECASE | re.DOTALL,
)
import re, ast

def _parse_assistant(assistant_text: str) -> dict:
    
    text = assistant_text or ""

    thinking = ""

    try:
        if '_THINKING_RE' in globals() and _THINKING_RE is not None:
            m_th = _THINKING_RE.search(text)
            if m_th:
                thinking = (m_th.group(1) if m_th.lastindex else m_th.group(0)).strip()
    except Exception:
        pass
    if not thinking:

        m = re.search(r'(?is)<think>\s*(.*?)\s*</think>', text)
        if m: thinking = m.group(1).strip()
    else:

        m = _FALLBACK_THINKING_RE.search(text)
        if m:
            thinking = m.group(1).strip()
    if not thinking:

        m = re.search(r'(?is)(?:^|\n)\s*(?:#\s*)?thinking\s*:?\s*(.*?)\s*(?=\n\s*(?:#\s*)?final\s*answer\b|$)', text)
        if m: thinking = m.group(1).strip()


    FA_BODY_RE = re.compile(
        r'(?is)\[(point|region)\s*,\s*(\[[\s\S]*?\])\s*,\s*([01](?:\.\d+)?)\s*\]'
    )
    mfa = FA_BODY_RE.search(text)
    if not mfa:

        try:
            if '_HDR_FINAL' in globals() and _HDR_FINAL is not None:
                mh = _HDR_FINAL.search(text)
                if mh:
                    mfa = FA_BODY_RE.search(text[mh.end():])
        except Exception:
            pass
    if not mfa:
        return {"mode": None, "confidence": None, "thinking": thinking}

    kind = mfa.group(1).lower()   
    body_str = mfa.group(2)         
    conf_str = mfa.group(3)


    try:
        body = ast.literal_eval(body_str)
    except Exception:
        return {"mode": None, "confidence": None, "thinking": thinking}

    try:
        conf = float(conf_str)
    except Exception:
        conf = None


    if kind == "point":
        if not isinstance(body, list):
            return {"mode": None, "confidence": conf, "thinking": thinking}
        return {"mode": "point", "point": body, "confidence": conf, "thinking": thinking}

    if kind == "region":

        if (isinstance(body, list) and len(body) == 2
                and isinstance(body[0], (list, tuple)) and isinstance(body[1], (list, tuple))):
            return {
                "mode": "region",
                "lb": list(body[0]),
                "ub": list(body[1]),
                "confidence": conf,
                "thinking": thinking,
            }
        return {"mode": None, "confidence": conf, "thinking": thinking}

    return {"mode": None, "confidence": conf, "thinking": thinking}


def _extract_param_names(prev_user_prompt: str) -> list[str] | None:
    m = _PARAM_ORDER_RE.search(prev_user_prompt)
    if not m:
        return None
    inside = m.group(1)
    return [s.strip() for s in inside.split(",")]

def _extract_existing_history(prev_user_prompt: str) -> list[str]:
    lines = prev_user_prompt.splitlines()
    hist, capture = [], False
    for ln in lines:
        if ln.strip().startswith("- Historical data"):
            capture = True
            continue
        if capture:
            if ln.startswith("  "):
                hist.append(ln.strip())     
            elif ln.strip().startswith("- "):
                break
    return hist



def _format_number(x: Any) -> str:
    if isinstance(x, (int, float)):
        return f"{x:.6g}"
    return str(x)

def _format_history_entry(
    param_names: Sequence[str] | None,
    executed_point: Sequence[Any] | dict | None,
    results: dict | None,
) -> str:
    if executed_point is None:
        params_str = "(executed_point=UNKNOWN)"
    else:
        if isinstance(executed_point, dict):
            if param_names:
                pairs = [f"{p}={_format_number(executed_point.get(p, 'NA'))}" for p in param_names]
            else:
                pairs = [f"{k}={_format_number(v)}" for k, v in executed_point.items()]
        else:
            if param_names and len(executed_point) == len(param_names):
                pairs = [f"{p}={_format_number(v)}" for p, v in zip(param_names, executed_point)]
            else:
                pairs = [f"x{i+1}={_format_number(v)}" for i, v in enumerate(executed_point)]
        params_str = "(" + ", ".join(pairs) + ")"

    if results:
        res_pairs = [f"{k}={_format_number(v)}" for k, v in results.items()]
        res_str = " → " + ", ".join(res_pairs)
    else:
        res_str = " → (no measured results)"

    return params_str + res_str

# ---------- Main builder (updated: no old_history by default) ----------

def build_next_user_prompt(
    make_user_prompt_fn: Callable[..., str],
    prev_user_prompt: str,
    assistant_text: str,
    *,
    executed_point: Sequence[Any] | dict | Iterable[Sequence[Any] | dict] | None,
    results: dict | Iterable[dict] | None,
    key_observations: str | None = None,
    this_round_focus: str | None = None,
    max_thinking_chars: int = 600,
    extra_history: list[str] | None = None,
    last_reasoning_override: str | None = None,
    include_old_history: bool = False,      
    history_keep_n: int | None = None,       
    **make_fn_kwargs,
) -> str:


    parsed = _parse_assistant(assistant_text)
    thinking_raw = (last_reasoning_override
                    if last_reasoning_override is not None
                    else parsed.get("thinking", "") or "")
    thinking = thinking_raw.strip()
    if thinking and max_thinking_chars is not None and max_thinking_chars >= 0:
        if len(thinking) > max_thinking_chars:
            thinking = thinking[:max_thinking_chars].rstrip() + " …"

    param_names = _extract_param_names(prev_user_prompt)

    old_history = []
    if include_old_history:
        old_history = _extract_existing_history(prev_user_prompt) or []
        if history_keep_n is not None and history_keep_n >= 0:
            old_history = old_history[:history_keep_n] 

    def _norm_points(obj):

        if obj is None:
            return []
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                return []

            if any(isinstance(e, (list, tuple, dict)) for e in obj):
                return list(obj)
            else:
                return [list(obj)]

        return [[obj]]

    def _norm_results(obj):

        if obj is None:
            return []
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                return []
            if all(isinstance(e, dict) for e in obj):
                return list(obj)
            else:
                return [{"value": e} for e in obj]
        return [{"value": obj}]

    ep_list = _norm_points(executed_point)   # list of point(dict | list)
    rs_list = _norm_results(results)         # list of dict


    k = max(len(ep_list), len(rs_list), 0)
    if k > 0:
        if len(ep_list) not in (0, k):
            if len(ep_list) == 1:
                ep_list = ep_list * k
            else:
                raise ValueError(f"[build_next_user_prompt] executed_point size mismatch: "
                                f"{len(ep_list)} vs results {len(rs_list)}")
        if len(rs_list) not in (0, k):
            if len(rs_list) == 1:
                rs_list = rs_list * k
            else:
                raise ValueError(f"[build_next_user_prompt] results size mismatch: "
                                f"{len(rs_list)} vs executed_point {len(ep_list)}")

    new_lines: list[str] = []
    if k > 0:

        for ep, rs in zip(ep_list, rs_list if rs_list else [None] * k):
            new_lines.append(_format_history_entry(param_names, ep, rs))

    history_newest_first = []
    if new_lines:
        history_newest_first.extend(new_lines)
    if extra_history:
        history_newest_first.extend(extra_history)
    if old_history:
        history_newest_first.extend(old_history)

    # ------------------------------------------------------
    # 5) hand off to dataset-specific prompt builder (make_)
    # ------------------------------------------------------
    next_user_prompt = make_user_prompt_fn(
        history=history_newest_first,                    # list[str], newest first
        last_reasoning=(thinking if thinking else None), 
        key_observations=key_observations,
        this_round_focus=this_round_focus,
        **make_fn_kwargs,
    )
    return next_user_prompt

