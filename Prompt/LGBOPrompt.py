"""Prompt helpers for LGBO.

Keep the format intentionally simple in V1 so the prompt layer remains easy to
test without requiring the full LGBO algorithm stack.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence


def build_lgbo_numeric_prompt(
    *,
    query_text: str,
    objective_name: str,
    param_specs: Sequence[Mapping[str, object]],
    history_lines: Iterable[str],
    previous_reasoning: str | None = None,
) -> str:
    lines = [
        "You are assisting a numeric hyperparameter optimization loop.",
        f"Objective: maximize {objective_name}.",
        "Return either:",
        "[point, [x1, ..., xd], confidence]",
        "or",
        "[region, [[lb1, ..., lbd], [ub1, ..., ubd]], confidence]",
        "",
        "Parameter order:",
    ]
    for spec in param_specs:
        lines.append(f"- {spec['name']}: [{spec['low']}, {spec['high']}] ({spec['kind']})")
    lines.append("")
    lines.append(f"Query: {query_text}")
    lines.append("History:")
    for item in history_lines:
        lines.append(f"- {item}")
    if previous_reasoning:
        lines.append("")
        lines.append(f"Previous reasoning: {previous_reasoning}")
    return "\n".join(lines)

