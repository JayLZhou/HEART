from __future__ import annotations

from typing import Any, Dict


class LGBOTraceStore:
    """Read/write LGBO-specific metadata on trial user attrs."""

    RAW_KEY = "lgbo_preference_raw"
    PARSED_KEY = "lgbo_preference_parsed"
    PLAN_KEY = "lgbo_plan"
    REASONING_KEY = "lgbo_reasoning"

    def write(self, trial: Any, *, raw: str | None, parsed: Dict[str, Any] | None, plan: Dict[str, Any] | None, reasoning: str | None) -> None:
        if raw is not None:
            trial.set_user_attr(self.RAW_KEY, raw)
        if parsed is not None:
            trial.set_user_attr(self.PARSED_KEY, parsed)
        if plan is not None:
            trial.set_user_attr(self.PLAN_KEY, plan)
        if reasoning is not None:
            trial.set_user_attr(self.REASONING_KEY, reasoning)

    def read_reasoning(self, trial: Any) -> str | None:
        return (getattr(trial, "user_attrs", {}) or {}).get(self.REASONING_KEY)

