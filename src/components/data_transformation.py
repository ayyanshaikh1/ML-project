import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.utils.utils import save_object
from config.config import DATA_TRANSFORMATION_CONFIG

logger = get_logger(__name__)

class DataTransformation:
    def __init__(self):
        """
        Initialize data transformation with configuration
        """
        self.transformation_config = DATA_TRANSFORMATION_CONFIG
    
    def get_data_transformer_object(self):
        """
        Create preprocessing pipeline for numerical features
        
        Returns:
            Sklearn Pipeline object
        """
        try:
            # Create preprocessing pipeline for numerical features
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            logger.info("Numerical preprocessing pipeline created")
            return numerical_pipeline
            
        except Exception as e:
            logger.error(f"Error in creating data transformer: {e}")
            raise CustomException(e)
    
    def initiate_data_transformation(self, train_file_path, test_file_path):
        """
        Initiate the data transformation process
        
        Args:
            train_file_path: Path to training data
            test_file_path: Path to test data
            
        Returns:
            Tuple of transformed train and test data arrays and preprocessor object path
        """
        try:
            # Create directories
            os.makedirs(self.transformation_config["transformed_train_dir"], exist_ok=True)
            os.makedirs(self.transformation_config["transformed_test_dir"], exist_ok=True)
            
            # Read data
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            
            logger.info(f"Train dataset shape: {train_df.shape}")
            logger.info(f"Test dataset shape: {test_df.shape}")
            
            # Separate target feature
            target_column = "Good/Bad"

            # Map target labels if they are strings like Good/Bad
            def map_target(series):
                if series.dtype == object:
                    mapping = {"Good": 1, "Bad": 0, "good": 1, "bad": 0, "GOOD": 1, "BAD": 0}
                    mapped = series.map(mapping)
                    if mapped.isna().all():
                        return series
                    return mapped
                return series

            if target_column in train_df.columns:
                train_df[target_column] = map_target(train_df[target_column])
            if target_column in test_df.columns:
                test_df[target_column] = map_target(test_df[target_column])

            # Build list of columns to drop from features
            drop_cols_train = [c for c in [target_column, "Wafer"] if c in train_df.columns]
            drop_cols_test = [c for c in [target_column, "Wafer"] if c in test_df.columns]

            input_feature_train_df = train_df.drop(columns=drop_cols_train, axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=drop_cols_test, axis=1)
            target_feature_test_df = test_df[target_column]
            
            # Select only numeric columns for preprocessing
            numeric_columns = input_feature_train_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_columns) == 0:
                raise CustomException("No numeric feature columns found for preprocessing.")

            input_feature_train_df = input_feature_train_df[numeric_columns]
            input_feature_test_df = input_feature_test_df[[c for c in numeric_columns if c in input_feature_test_df.columns]]
            # Ensure test has all train numeric columns (fill missing with 0)
            for col in numeric_columns:
                if col not in input_feature_test_df.columns:
                    input_feature_test_df[col] = 0
            # Reorder test columns to match train
            input_feature_test_df = input_feature_test_df[numeric_columns]

            # Get preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Transform features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # Combine features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            # Save transformed data
            transformed_train_file = os.path.join(self.transformation_config["transformed_train_dir"], "transformed_train.npz")
            transformed_test_file = os.path.join(self.transformation_config["transformed_test_dir"], "transformed_test.npz")
            
            np.savez_compressed(transformed_train_file, data=train_arr)
            np.savez_compressed(transformed_test_file, data=test_arr)
            
            logger.info(f"Transformed train data saved at: {transformed_train_file}")
            logger.info(f"Transformed test data saved at: {transformed_test_file}")
            
            # Save preprocessing object along with selected numeric columns
            preprocessor_file_path = self.transformation_config["preprocessed_object_file_path"]
            save_object(preprocessor_file_path, {
                "pipeline": preprocessing_obj,
                "numeric_columns": numeric_columns
            })
            
            logger.info(f"Preprocessor object saved at: {preprocessor_file_path}")
            
            return (
                train_arr,
                test_arr,
                preprocessor_file_path
            )
            
        except Exception as e:
            logger.error(f"Error in data transformation: {e}")
            raise CustomException(e)