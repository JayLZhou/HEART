"""Prompt helpers for LGBO."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence


LGBO_NUMERIC_SYSTEM_PROMPT = """You are assisting a hyperparameter optimization loop.

Return exactly two blocks:

Thinking:
- Briefly explain which direction or region looks promising given the query and trial history.

Final Answer:
- Return exactly one of:
  [point, [x1, ..., xd], confidence]
  [region, [[lb1, ..., lbd], [ub1, ..., ubd]], confidence]

Rules:
- confidence must be in [0, 1]
- follow the declared parameter order exactly
- keep all numeric values inside the provided bounds
- for categorical string values, use quoted Python string literals
- for boolean values, use Python literals True / False
- for categorical values in region mode, repeat the same literal in the lower and upper lists for that dimension
- prefer region when uncertainty is high or a neighborhood appears promising
- prefer point when the evidence supports a sharp recommendation
"""


def build_lgbo_numeric_prompt(
    *,
    query_text: str,
    objective_name: str,
    param_specs: Sequence[Mapping[str, object]],
    history_lines: Iterable[str],
    previous_reasoning: str | None = None,
) -> str:
    lines = [
        f"Objective: maximize {objective_name}.",
        "Task:",
        f"- {query_text}",
        "Return either:",
        "[point, [x1, ..., xd], confidence]",
        "or",
        "[region, [[lb1, ..., lbd], [ub1, ..., ubd]], confidence]",
        "",
        "Parameter order:",
    ]
    for spec in param_specs:
        if spec["kind"] in {"int", "float"}:
            lines.append(f"- {spec['name']}: [{spec['low']}, {spec['high']}] ({spec['kind']})")
        else:
            lines.append(f"- {spec['name']}: choices={list(spec['choices'])} ({spec['kind']})")
    lines.append("")
    lines.append("Completed trial history:")
    has_history = False
    for item in history_lines:
        has_history = True
        lines.append(f"- {item}")
    if not has_history:
        lines.append("- No completed trials yet.")
    if previous_reasoning:
        lines.append("")
        lines.append(f"Previous reasoning: {previous_reasoning}")
    return "\n".join(lines)

