import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import urllib.request
from src.utils.logger import get_logger
from src.utils.exception import CustomException
from config.config import DATA_INGESTION_CONFIG

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self):
        """
        Initialize data ingestion with configuration
        """
        self.ingestion_config = DATA_INGESTION_CONFIG
        
    def initiate_data_ingestion(self, file_path=None):
        """
        Initiate the data ingestion process
        
        Args:
            file_path: Path to the input data file
            
        Returns:
            Tuple of paths to train and test data
        """
        try:
            # Create directories
            os.makedirs(self.ingestion_config["ingested_train_dir"], exist_ok=True)
            os.makedirs(self.ingestion_config["ingested_test_dir"], exist_ok=True)
            
            logger.info("Data ingestion initiated")
            
            # If file path is not provided, prefer a labeled file (contains target column)
            if file_path is None:
                raw_dir = self.ingestion_config["raw_data_dir"]
                os.makedirs(raw_dir, exist_ok=True)
                raw_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir)]
                # If empty on Render, optionally download from DATA_FILE_URL
                if len(raw_files) == 0:
                    download_url = self.ingestion_config.get("dataset_download_url")
                    if download_url:
                        target_path = os.path.join(raw_dir, "dataset.csv")
                        logger.info(f"Raw data folder empty. Downloading dataset from {download_url} -> {target_path}")
                        urllib.request.urlretrieve(download_url, target_path)
                        raw_files = [target_path]
                    else:
                        raise Exception("No files found in raw data directory and no DATA_FILE_URL provided")

                def has_target(fp):
                    try:
                        if fp.lower().endswith('.csv'):
                            df_head = pd.read_csv(fp, nrows=5)
                        elif fp.lower().endswith(('.xls', '.xlsx')):
                            df_head = pd.read_excel(fp, nrows=5)
                        else:
                            return False
                        return "Good/Bad" in df_head.columns
                    except Exception:
                        return False

                labeled = [f for f in raw_files if has_target(f)]
                candidate_files = labeled if labeled else raw_files
                # Choose the most recently modified candidate
                file_path = max(candidate_files, key=lambda p: os.path.getmtime(p))
            
            logger.info(f"Reading data from {file_path}")
            
            # Read data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            else:
                raise Exception(f"Unsupported file format: {file_path}")
            
            logger.info(f"Dataset shape: {df.shape}")
            
            # Split data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save train and test data
            train_file_path = os.path.join(self.ingestion_config["ingested_train_dir"], "train.csv")
            test_file_path = os.path.join(self.ingestion_config["ingested_test_dir"], "test.csv")
            
            train_set.to_csv(train_file_path, index=False)
            test_set.to_csv(test_file_path, index=False)
            
            logger.info(f"Train data saved at: {train_file_path}")
            logger.info(f"Test data saved at: {test_file_path}")
            
            return train_file_path, test_file_path
            
        except Exception as e:
            logger.error(f"Error in data ingestion: {e}")
            raise CustomException(e)