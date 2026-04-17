"""Prompt helpers for LGBO.

This module intentionally reuses the original prompt style under ``lgbo/prompt.py``.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

try:
    from lgbo.prompt import SYSTEM_PROMPT as _LGBO_BASE_SYSTEM_PROMPT
    from lgbo.prompt import make_user_prompt as _make_user_prompt
except Exception:
    _LGBO_BASE_SYSTEM_PROMPT = None
    _make_user_prompt = None


LGBO_SYSTEM_PROMPT = _LGBO_BASE_SYSTEM_PROMPT or """You are assisting a hyperparameter optimization loop.

Return exactly two blocks:

Thinking:
- Briefly explain which direction or region looks promising given the query and trial history.

Final Answer:
- Return exactly one of:
  [point, [x1, ..., xd], confidence]
  [region, [[lb1, ..., lbd], [ub1, ..., ubd]], confidence]
"""


def _parameter_label(spec: Mapping[str, object]) -> str:
    kind = spec.get("kind")
    if kind in {"float", "int"}:
        return f"{spec['name']} ({kind}) in [{spec['low']}, {spec['high']}]"
    return f"{spec['name']} ({kind}) choices={spec.get('choices')}"


def _shared_region_text(shared_region: Mapping[str, object] | None) -> str | None:
    if not shared_region:
        return None
    parts = [f"mode={shared_region.get('mode', 'none')}"]
    if shared_region.get("lower") is not None:
        parts.append(f"lower={shared_region.get('lower')}")
    if shared_region.get("upper") is not None:
        parts.append(f"upper={shared_region.get('upper')}")
    if shared_region.get("point") is not None:
        parts.append(f"point={shared_region.get('point')}")
    if shared_region.get("confidence") is not None:
        parts.append(f"confidence={shared_region.get('confidence')}")
    return "; ".join(parts)


def build_lgbo_prompt(
    *,
    query_text: str,
    objective_name: str,
    param_specs: Sequence[Mapping[str, object]],
    history_lines: Iterable[str],
    previous_reasoning: str | None = None,
    round_id: int | None = None,
    cluster_id: int | None = None,
    shared_region: Mapping[str, object] | None = None,
) -> str:
    history_items = list(history_lines)
    param_labels = [_parameter_label(spec) for spec in param_specs]
    focus_items = []
    if round_id is not None:
        focus_items.append(f"round={round_id}")
    if cluster_id is not None:
        focus_items.append(f"cluster_id={cluster_id}")
    if shared_region:
        focus_items.append(f"shared_region={_shared_region_text(shared_region)}")
    focus_text = " | ".join(focus_items) if focus_items else None
    constraints = (
        "Return exactly one format: "
        "[point, [x1,...,xd], confidence] or [region, [[lb1,...,lbd],[ub1,...,ubd]], confidence]. "
        "Respect declared parameter order and bounds/choices."
    )

    if _make_user_prompt is not None:
        prompt = _make_user_prompt(
            background="Hyperparameter tuning for a retrieval-augmented generation pipeline.",
            parameters=param_labels,
            objective=f"Maximize {objective_name}. Task/query: {query_text}",
            constraints=constraints,
            history=history_items or None,
            last_reasoning=previous_reasoning,
            key_observations=None,
            this_round_focus=focus_text,
        )
    else:
        lines = [
            f"Objective: maximize {objective_name}.",
            f"Task: {query_text}",
            "Parameter order:",
            *[f"- {label}" for label in param_labels],
        ]
        if history_items:
            lines.append("Completed trial history:")
            lines.extend(f"- {item}" for item in history_items)
        if previous_reasoning:
            lines.append(f"Previous reasoning: {previous_reasoning}")
        if focus_text:
            lines.append(f"This round focus: {focus_text}")
        lines.append(constraints)
        prompt = "\n".join(lines)

    # Compatibility footer for existing parsers/tests that look for these markers.
    compat_lines = ["", f"Round: {round_id}" if round_id is not None else "Round: -1"]
    if cluster_id is not None:
        compat_lines.append(f"Cluster ID: {cluster_id}")
    if shared_region:
        compat_lines.append("")
        compat_lines.append("Shared cluster region:")
        compat_lines.append(f"- mode={shared_region.get('mode', 'none')}")
        if shared_region.get("lower") is not None:
            compat_lines.append(f"- lower={shared_region.get('lower')}")
        if shared_region.get("upper") is not None:
            compat_lines.append(f"- upper={shared_region.get('upper')}")
        if shared_region.get("point") is not None:
            compat_lines.append(f"- point={shared_region.get('point')}")
        if shared_region.get("confidence") is not None:
            compat_lines.append(f"- confidence={shared_region.get('confidence')}")
    compat_lines.append("")
    compat_lines.append("Completed trial history:")
    if history_items:
        compat_lines.extend(f"- {item}" for item in history_items)
    else:
        compat_lines.append("- No completed trials yet.")
    compat_lines.append("Task:")
    compat_lines.append(f"- {query_text}")
    compat_lines.append("Parameter order:")
    for spec in param_specs:
        if spec["kind"] in {"float", "int"}:
            compat_lines.append(f"- {spec['name']}: [{spec['low']}, {spec['high']}] ({spec['kind']})")
        else:
            compat_lines.append(f"- {spec['name']}: choices={spec['choices']} ({spec['kind']})")
    return prompt + "\n" + "\n".join(compat_lines)


LGBO_NUMERIC_SYSTEM_PROMPT = LGBO_SYSTEM_PROMPT
build_lgbo_numeric_prompt = build_lgbo_prompt
