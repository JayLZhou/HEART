import os
import joblib

from Storage.NameSpace import Namespace

class OptunaStorage:
    def __init__(self, namespace: Namespace):
        self.namespace = namespace
 


    
    def save_study(self, study, file_path: str = None) -> None:
        """Save complete Optuna study object to file using joblib
        
        This preserves the complete study state, allowing you to resume optimization later.
        
        Args:
            study: Optuna Study object to save
            file_path: Optional file path. If None, will use namespace to generate path
        """
        if file_path is None:
            # Generate file path from namespace
            save_path = self.namespace.get_save_path("optuna_study")
            file_path = os.path.join(save_path, "study.pkl")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save complete study object using joblib
        joblib.dump(study, file_path)
    
    def load_study(self, file_path: str = None):
        """Load complete Optuna study object from file using joblib
        
        Args:
            file_path: Optional file path. If None, will use namespace to generate path
            
        Returns:
            Optuna Study object, or None if file doesn't exist
        """
        if file_path is None:
            load_path = self.namespace.get_load_path("optuna_study")
            if load_path is None:
                return None
            file_path = os.path.join(load_path, "study.pkl")
        
        if not os.path.exists(file_path):
            return None
        
        # Load complete study object using joblib
        return joblib.load(file_path)