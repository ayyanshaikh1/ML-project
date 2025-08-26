import os
import sys
import pandas as pd
import numpy as np
from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.utils.utils import load_object

logger = get_logger(__name__)

class PredictionPipeline:
    def __init__(self, model_path=None, preprocessor_path=None):
        """
        Initialize prediction pipeline
        
        Args:
            model_path: Path to trained model file
            preprocessor_path: Path to preprocessor object file
        """
        try:
            # If paths are not provided, try to use the latest artifacts; otherwise fall back to a safe dummy model
            if model_path is None or preprocessor_path is None:
                artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")

                latest_dir = None
                if os.path.isdir(artifacts_dir):
                    timestamp_dirs = [d for d in os.listdir(artifacts_dir) if os.path.isdir(os.path.join(artifacts_dir, d))]
                    if timestamp_dirs:
                        latest_dir = max(timestamp_dirs)

                if latest_dir is not None:
                    if model_path is None:
                        model_path = os.path.join(artifacts_dir, latest_dir, "model_trainer", "model.pkl")
                    if preprocessor_path is None:
                        preprocessor_path = os.path.join(artifacts_dir, latest_dir, "data_transformation", "preprocessed.pkl")

            if model_path and preprocessor_path and os.path.isfile(model_path) and os.path.isfile(preprocessor_path):
                logger.info(f"Loading model from: {model_path}")
                logger.info(f"Loading preprocessor from: {preprocessor_path}")
                self.model = load_object(model_path)
                preprocessor_bundle = load_object(preprocessor_path)
                if isinstance(preprocessor_bundle, dict) and "pipeline" in preprocessor_bundle:
                    self.preprocessor = preprocessor_bundle["pipeline"]
                    self.numeric_columns = preprocessor_bundle.get("numeric_columns")
                else:
                    self.preprocessor = preprocessor_bundle
                    self.numeric_columns = None
                self.using_fallback = False
            else:
                # Fallback: no trained artifacts available
                logger.warning("No trained model/preprocessor found. Using fallback predictor. Upload a dataset to /train to build a real model.")

                class _IdentityPreprocessor:
                    def transform(self, X):
                        return X.values if hasattr(X, "values") else np.asarray(X)

                class _MajorityGoodModel:
                    def predict(self, X):
                        # Predict 'Good' (1) for all rows as a benign default
                        n = len(X) if hasattr(X, "__len__") else 1
                        return np.ones(n, dtype=int)

                self.preprocessor = _IdentityPreprocessor()
                self.model = _MajorityGoodModel()
                self.using_fallback = True

        except Exception as e:
            logger.error(f"Error initializing prediction pipeline: {e}")
            raise CustomException(e)
    
    def predict(self, features):
        """
        Make predictions on input features
        
        Args:
            features: Input features as DataFrame or array
            
        Returns:
            Predictions array
        """
        try:
            logger.info("Starting prediction")
            
            # Preprocess features
            if isinstance(features, pd.DataFrame):
                # Remove Wafer column if present
                if "Wafer" in features.columns:
                    wafer_ids = features["Wafer"].values
                    features = features.drop(columns=["Wafer"])
                
                # Remove target column if present
                if "Good/Bad" in features.columns:
                    features = features.drop(columns=["Good/Bad"])
            
            # Restrict to numeric columns learned during training if available
            if isinstance(features, pd.DataFrame):
                if hasattr(self, "numeric_columns") and self.numeric_columns:
                    for col in self.numeric_columns:
                        if col not in features.columns:
                            features[col] = 0
                    features = features[self.numeric_columns]
            transformed_features = self.preprocessor.transform(features)
            
            # Make predictions
            predictions = self.model.predict(transformed_features)
            
            logger.info("Prediction completed successfully")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise CustomException(e)


class CustomData:
    """
    Class to convert user input data to DataFrame for prediction
    """
    def __init__(self, **kwargs):
        """
        Initialize with sensor values
        """
        self.sensor_data = kwargs
    
    def get_data_as_dataframe(self):
        """
        Convert input data to DataFrame
        
        Returns:
            Pandas DataFrame
        """
        try:
            return pd.DataFrame(self.sensor_data, index=[0])
        except Exception as e:
            logger.error(f"Error converting data to DataFrame: {e}")
            raise CustomException(e)