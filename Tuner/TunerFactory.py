"""
Tuner Factory for creating different types of tuners.

Supports:
- BO-based (Bayesian Optimization): Optuna-based tuners
- MAB-based (Multi-Armed Bandit): Bandit-based tuners
- Others: Other optimization methods
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from Common.BaseFactory import GenericFactory
from Common.Logger import logger
from Tuner.BasicTuner import BasicTuner
from Tuner.BOTuner.OptunaTuner import OptunaTuner
from Common.Constants import TunerType



class TunerFactory(GenericFactory):
    """Factory for creating tuner instances based on type."""

    def __init__(self):
        creators = {
            TunerType.BO: self._create_bo_tuner,
            TunerType.MAB: self._create_mab_tuner,
            TunerType.OTHER: self._create_other_tuner,
            
        }
        super().__init__(creators)

    def get_tuner(
        self,
        tuner_type: TunerType | str,
        config=None,
        **kwargs
    ) -> BasicTuner:
        """
        Get a tuner instance by type.
        
        Args:
            tuner_type: Tuner type (TunerType)
            config: Configuration object
            **kwargs: Additional arguments for tuner creation
            
        Returns:
            BasicTuner: Tuner instance
    
        """
     
        return super().get_instance(tuner_type, config=config, **kwargs)

    def _create_bo_tuner(self, config=None, **kwargs) -> BasicTuner:
        """Create a Bayesian Optimization based tuner (Optuna)."""
        if config is None:
            raise ValueError("study_config is required for BO tuner")
        
        logger.info("Creating BO-based tuner (Optuna)")
        return OptunaTuner(config=config, **kwargs)

    def _create_mab_tuner(self, study_config=None, **kwargs) -> BasicTuner:
        """Create a Multi-Armed Bandit based tuner."""
        try:
            from Tuner.MABTuner.MABTuner import MABTuner
        except ImportError:
            logger.warning(
                "MABTuner not found. Creating a placeholder MAB tuner. "
                "Please implement MABTuner in Tuner/MABTuner/MABTuner.py"
            )
            # Return a placeholder that implements BasicTuner interface
            return PlaceholderMABTuner(study_config=study_config, **kwargs)
        
        if study_config is None:
            raise ValueError("study_config is required for MAB tuner")
        
        logger.info("Creating MAB-based tuner")
        return MABTuner(study_config=study_config, **kwargs)

    def _create_other_tuner(self, study_config=None, **kwargs) -> BasicTuner:
        """Create other type of tuner (e.g., random search, grid search)."""
        # Try to get tuner type from kwargs or config
        other_type = kwargs.pop("other_type", None) or (
            getattr(study_config, "tuner_type", None) if study_config else None
        )
        
        if other_type == "random" or other_type is None:
            try:
                from Tuner.OtherTuner.RandomTuner import RandomTuner
                logger.info("Creating Random Search tuner")
                return RandomTuner(study_config=study_config, **kwargs)
            except ImportError:
                logger.warning("RandomTuner not found, using placeholder")
                return PlaceholderOtherTuner(
                    study_config=study_config, 
                    tuner_type="random",
                    **kwargs
                )
        elif other_type == "grid":
            try:
                from Tuner.OtherTuner.GridTuner import GridTuner
                logger.info("Creating Grid Search tuner")
                return GridTuner(study_config=study_config, **kwargs)
            except ImportError:
                logger.warning("GridTuner not found, using placeholder")
                return PlaceholderOtherTuner(
                    study_config=study_config,
                    tuner_type="grid",
                    **kwargs
                )
        else:
            logger.warning(f"Unknown other tuner type: {other_type}, using placeholder")
            return PlaceholderOtherTuner(
                study_config=study_config,
                tuner_type=other_type,
                **kwargs
            )

    def _raise_for_key(self, key: Any):
        import pdb
        pdb.set_trace()
        raise ValueError(
            f"Unknown tuner type: {key}. "
            f"Supported types: {', '.join([t.value for t in TunerType])}"
        )


# Placeholder implementations for missing tuners
class PlaceholderMABTuner(BasicTuner):
    """Placeholder MAB tuner implementation."""
    
    def __init__(self, study_config=None, **kwargs):
        super().__init__()
        self.study_config = study_config
        logger.warning("Using placeholder MAB tuner. Please implement MABTuner.")

    def _create_tuner(self):
        raise NotImplementedError("MABTuner not implemented yet")

    def objective(self):
        raise NotImplementedError("MABTuner not implemented yet")

    def evaluate(self):
        raise NotImplementedError("MABTuner not implemented yet")

    def get_sampler(self):
        raise NotImplementedError("MABTuner not implemented yet")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("MABTuner not implemented yet")

    def save_config(self, *args, **kwargs):
        raise NotImplementedError("MABTuner not implemented yet")


class PlaceholderOtherTuner(BasicTuner):
    """Placeholder for other tuner types."""
    
    def __init__(self, study_config=None, tuner_type="random", **kwargs):
        super().__init__()
        self.study_config = study_config
        self.tuner_type = tuner_type
        logger.warning(f"Using placeholder tuner for type: {tuner_type}")

    def _create_tuner(self):
        raise NotImplementedError(f"{self.tuner_type} tuner not implemented yet")

    def objective(self):
        raise NotImplementedError(f"{self.tuner_type} tuner not implemented yet")

    def evaluate(self):
        raise NotImplementedError(f"{self.tuner_type} tuner not implemented yet")

    def get_sampler(self):
        raise NotImplementedError(f"{self.tuner_type} tuner not implemented yet")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(f"{self.tuner_type} tuner not implemented yet")

    def save_config(self, *args, **kwargs):
        raise NotImplementedError(f"{self.tuner_type} tuner not implemented yet")


# Factory instance
_tuner_factory = TunerFactory()


def get_tuner(
    config,
    **kwargs
) -> BasicTuner:
    """
    Convenience function to get a tuner instance.
    
    Args:
        tuner_type: Tuner type (TunerType enum or string: "bo", "mab", "other")
        study_config: Study configuration object
        **kwargs: Additional arguments for tuner creation
        
    Returns:
        BasicTuner: Tuner instance
        
    Examples:
        >>> tuner = get_tuner("bo", study_config=config)
        >>> tuner = get_tuner(TunerType.MAB, study_config=config)
    """

    return _tuner_factory.get_tuner(config.tuner_type, config=config, **kwargs)

