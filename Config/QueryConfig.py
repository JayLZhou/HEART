#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Optional, List
from pydantic import BaseModel, Field
from Config.SearchSpaceMix import *
import typing as T
import math
from Common.Constants import DEFAULT_LLMS


class QueryConfig(BaseModel, SearchSpaceMixin):
    """Query configuration"""
    subquestion_engine_llms: T.List[str] = Field(
        default_factory=lambda: DEFAULT_LLMS,
        description="LLMs for the sub-question engine.",
    )
    subquestion_response_synthesizer_llms: T.List[str] = Field(
        default_factory=lambda: DEFAULT_LLMS,
        description="LLMs for synthesizing responses to subquestions.",
    )

    def defaults(self, prefix: str = "") -> T.Dict[str, T.Any]:
        return {
            f"{prefix}subquestion_engine_llm": self.subquestion_engine_llms[0],
            f"{prefix}subquestion_response_synthesizer_llm": self.subquestion_response_synthesizer_llms[
                0
            ],
        }

    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        return {
            f"{prefix}subquestion_engine_llm": CategoricalDistribution(
                self.subquestion_engine_llms
            ),
            f"{prefix}subquestion_response_synthesizer_llm": CategoricalDistribution(
                self.subquestion_response_synthesizer_llms
            ),
        }

    def get_cardinality(self) -> int:
        categorical_dists = [
            self.subquestion_engine_llms,
            self.subquestion_response_synthesizer_llms,
        ]
        return math.prod([len(dist) for dist in categorical_dists])
