
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class StudyConfig(BaseSettings):
    name: str = Field(description="Name of the Optuna study.")
    dataset: T.Annotated[  # type: ignore
        T.Union[
            *SyftrQADataset.__subclasses__(),  # type: ignore
            *HotPotQAHF.__subclasses__(),  # type: ignore
            *FinanceBenchHF.__subclasses__(),  # type: ignore
            *CragTask3HF.__subclasses__(),  # type: ignore
            *DRDocsHF.__subclasses__(),  # type: ignore
        ],
        Field(discriminator="xname"),
    ] = Field(description="Dataset configuration.")
    evaluation: Evaluation = Field(
        default_factory=Evaluation, description="LLM-as-a-judge configuration."
    )
    reuse_study: bool = Field(
        default=True, description="Whether to reuse an existing study."
    )
    recreate_study: bool = Field(
        default=True,
        description="Whether to recreate the study if it already exists (potentially deleting old data).",
    )
    search_space: SearchSpace = Field(
        default_factory=SearchSpace,
        description="Search space configuration for the optimization.",
    )
    optimization: OptimizationConfig = Field(
        default_factory=OptimizationConfig,
        description="Optimization process configuration.",
    )


    timeouts: TimeoutConfig = Field(
        default_factory=TimeoutConfig,
        description="Timeout configurations for various stages.",
    )
    toy_mode: bool = Field(
        default=False, description="Whether to run in toy mode (with smaller dataset)."
    )

    model_config = SettingsConfigDict(
        extra="forbid",  # Forbids unknown fields
        yaml_file=cfg.study_config_file or Path("Idontexist"),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: T.Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> T.Tuple[PydanticBaseSettingsSource, ...]:
        """Study config can be loaded from a yaml file.

        Use SYFTR_STUDY_CONFIG_FILE env var or
        'study_config_file: <path> in the top-level of config.yaml
        to choose a study config file, or use the from_file factory method.

        Parameters passed to StudyConfig.__init__ will take precedence
        over the yaml file.
        """
        if cfg.study_config_file and not cfg.study_config_file.exists():
            raise ValueError(
                f"Study configuration file cannot be found at {cfg.study_config_file.resolve()}"
            )

        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    @classmethod
    def from_file(cls, path: Path | str, *args, **kwargs) -> "StudyConfig":
        """Use from_file to load from a given config file path.

        *args and **kwargs are the same as the StudyConfig constructor
        and take precedence over values loaded from the config file.

        cfg.study_config_file is ignored when this method is used.
        """
        if not Path(path).exists():
            raise ValueError(
                f"Study configuration file cannot be found at {Path(path).resolve()}"
            )

        klass = deepcopy(cls)
        _orig = klass.model_config.pop("yaml_file", None)
        klass.model_config = SettingsConfigDict(**cls.model_config, yaml_file=path)
        instance = klass(*args, **kwargs)
        klass.model_config["yaml_file"] = _orig
        return instance

    def replace_llm_name(self, params: T.Dict[str, T.Any]):
        """
        Replace the LLM name in the params with the replacement_llm_name.
        With this functionality, we can easily run historical flows with a different LLM.
        """
        assert self.pareto, "No Pareto config is set"
        assert self.pareto.replacement_llm_name, "No replacement LLM name is set"

        replacement_llm_name = self.pareto.replacement_llm_name
        params["response_synthesizer_llm"] = replacement_llm_name
