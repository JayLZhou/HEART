from typing import Optional, List

from pydantic import BaseModel, ConfigDict, Field, model_validator

from Option.Config2 import Config
from Common.Context import Context
from Provider.BaseLLM import BaseLLM


class ContextMixin(BaseModel):
    """Mixin class for context and config"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Pydantic has bug on _private_attr when using inheritance, so we use private_* instead
    # - https://github.com/pydantic/pydantic/issues/7142
    # - https://github.com/pydantic/pydantic/issues/7083
    # - https://github.com/pydantic/pydantic/issues/7091

    private_context: Optional[Context] = Field(default=None, exclude=True)
    private_config: Optional[Config] = Field(default=None, exclude=True)

    private_llm: Optional[BaseLLM] = Field(default=None, exclude=True)
    private_llms: Optional[List[BaseLLM]] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def validate_context_mixin_extra(self):

        self._process_context_mixin_extra()
        return self

    def _process_context_mixin_extra(self):
        """Process the extra field"""
        kwargs = self.model_extra or {}

        self.set_context(kwargs.pop("context", None))
        self.set_config(kwargs.pop("config", None))
        self.set_llm(kwargs.pop("llm", None))
        self.set_llms(kwargs.pop("llms", None))

    def set(self, k, v, override=False):
        """Set attribute"""
        if override or not self.__dict__.get(k):
            self.__dict__[k] = v

    def set_context(self, context: Context, override=True):
        """Set context"""
        self.set("private_context", context, override)

    def set_config(self, config: Config, override=False):
        """Set config"""
        self.set("private_config", config, override)
        if config is not None:
            _ = self.llm  # init llm
            _ = self.llms  # init llms

    def set_llm(self, llm: BaseLLM, override=False):
        """Set llm"""
        self.set("private_llm", llm, override)

    def set_llms(self, llms: List[BaseLLM], override=False):
        """Set llms"""
        self.set("private_llms", llms, override)

    @property
    def config(self) -> Config:
        """Role config: role config > context config"""
        if self.private_config:
            return self.private_config
        return self.context.config

    @config.setter
    def config(self, config: Config) -> None:
        """Set config"""
        self.set_config(config)

    @property
    def context(self) -> Context:
        """Role context: role context > context"""
        if self.private_context:
            return self.private_context
        return Context()

    @context.setter
    def context(self, context: Context) -> None:
        """Set context"""
        self.set_context(context)

    @property
    def llm(self) -> BaseLLM:
        """Role llm: if not existed, init from role.config
        Note: for multiple llms, only the first one is used for setting llm
        """
        if not self.private_llm:        
            self.private_llm = self.context.llm_with_cost_manager_from_llm_config(self.config.llms[0])
        return self.private_llm

    @property
    def llms(self) -> List[BaseLLM]:
        """Role llms: if not existed, init from role.config
        Note: for multiple llms, only the first one is used for setting llm
        """
        self.private_llms = [self.context.llm_with_cost_manager_from_llm_config(llm) for llm in self.config.llms]

        return self.private_llms

    @llms.setter
    def llms(self, llms: List[BaseLLM]) -> None:
        """Set llms"""

        self.private_llms = llms
    @llm.setter
    def llm(self, llm: BaseLLM) -> None:
        """Set llm"""
        self.private_llm = llm

    def get_llm(self, model_name: str) -> BaseLLM:
        """ 
        Get a LLM instance based on the model name.
        Args:
            model_name: LLM model name
            
        Returns:
            BaseLLM instance
        """

        if not hasattr(self, "private_llm_map"):
            # build the map of model name to llm
            self.private_llm_map = {llm.model: llm for llm in self.private_llms}
        if model_name not in self.private_llm_map:
            raise ValueError(f"Model '{model_name}' not found in LLM pool. Available models: {list(self.private_llm_map.keys())}")
        return self.private_llm_map[model_name]