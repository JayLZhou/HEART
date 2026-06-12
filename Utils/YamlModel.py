from pathlib import Path
from typing import Dict, Optional
import os
import re

import yaml
from pydantic import BaseModel, model_validator


ENV_VAR_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*))?\}$")


class YamlModel(BaseModel):
    """Base class for yaml model"""

    extra_fields: Optional[Dict[str, str]] = None

    @classmethod
    def read_yaml(cls, file_path: Path, encoding: str = "utf-8") -> Dict:
        """Read yaml file and return a dict"""
        if not file_path.exists():
            return {}
        with open(file_path, "r", encoding=encoding) as file:
            data = yaml.safe_load(file) or {}
        return cls.resolve_env_vars(data)

    @classmethod
    def resolve_env_vars(cls, value):
        """Resolve ${ENV_VAR} and ${ENV_VAR:-default} values recursively."""
        if isinstance(value, dict):
            return {k: cls.resolve_env_vars(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls.resolve_env_vars(item) for item in value]
        if isinstance(value, str):
            match = ENV_VAR_PATTERN.match(value)
            if not match:
                return value

            env_key, default = match.groups()
            env_value = os.getenv(env_key, default)
            if env_value is None:
                raise ValueError(f"Environment variable '{env_key}' is not set")
            return env_value
        return value

   
class YamlModelWithoutDefault(YamlModel):
    """YamlModel without default values"""

    @model_validator(mode="before")
    @classmethod
    def check_not_default_config(cls, values):
        """Check if there is any default config in config2.yaml"""
        if any(["YOUR" in v for v in values]):
            raise ValueError("Please set your config in config2.yaml")
        return values