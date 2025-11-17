import os
import joblib
import optuna

from Storage.NameSpace import Namespace

class OptunaStorage:
    def __init__(self, namespace: Namespace):
        self.namespace = namespace
    
    # def save_study(self, study, file_path: str = None) -> None:
    #     """Save complete Optuna study object to file using joblib
        
    #     This preserves the complete study state, allowing you to resume optimization later.
        
    #     Args:
    #         study: Optuna Study object to save
    #         file_path: Optional file path. If None, will use namespace to generate path
    #     """
    #     if file_path is None:
    #         # Generate file path from namespace
    #         save_path = self.namespace.get_save_path("optuna_study")
    #         file_path = os.path.join(save_path, "study.db")
        
    #     # Ensure directory exists
    #     os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    def load_study(self, study_name: str, file_path: str = None):
        """Load complete Optuna study object from file using joblib
        
        Args:
            file_path: Optional file path. If None, will use namespace to generate path
            
        Returns:
            Optuna Study object, or None if file doesn't exist
        """
        storage = self.get_storage(file_path)
        
        # Load complete study object using joblib
        study = optuna.load(
            study_name=study_name,
            storage=storage,
        )
        return study


    def get_storage(self) -> str:
        save_path = self.namespace.workspace.get_save_path()
        save_path = os.path.join(save_path, "optuna_study")
        file_path = os.path.join(save_path, "study.db")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        storage = f"sqlite:///{file_path}"
        return storage