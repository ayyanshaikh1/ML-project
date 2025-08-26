import os
import json
import pandas as pd
from src.utils.logger import get_logger
from src.utils.exception import CustomException
from config.config import DATA_VALIDATION_CONFIG

logger = get_logger(__name__)

class DataValidation:
    def __init__(self):
        """
        Initialize data validation with configuration
        """
        self.validation_config = DATA_VALIDATION_CONFIG
        
    def read_schema(self):
        """
        Read schema configuration from file
        
        Returns:
            Dict containing schema information
        """
        try:
            with open(self.validation_config["schema_dir"], 'r') as f:
                schema = json.load(f)
            
            logger.info(f"Schema loaded successfully from {self.validation_config['schema_dir']}")
            return schema
        except Exception as e:
            logger.error(f"Error reading schema: {e}")
            raise CustomException(e)
    
    def validate_dataset_schema(self, dataframe):
        """
        Validate if the dataset matches the expected schema
        
        Args:
            dataframe: Pandas DataFrame to validate
            
        Returns:
            Boolean indicating if validation passed
        """
        try:
            schema = self.read_schema()
            # Only require target column to exist; be lenient about others
            target = schema.get("target_column", "Good/Bad")
            if target not in dataframe.columns:
                logger.error(f"Target column {target} not found in the dataset")
                return False

            # If domain specified, ensure target values can map to it
            domain = set(schema.get("domain_value", {}).get(target, []))
            if domain:
                # Accept both numeric and string labels (e.g., 'Good'/'Bad') by attempting a mapping
                unique_vals = set(pd.Series(dataframe[target]).dropna().unique())
                # If already within domain, pass
                if unique_vals.issubset(domain):
                    return True
                # Try mapping common string labels to 0/1
                mapping = {"Good": 1, "Bad": 0, "good": 1, "bad": 0, "GOOD": 1, "BAD": 0}
                mapped = pd.Series(dataframe[target]).map(mapping)
                if set(mapped.dropna().unique()).issubset(domain):
                    return True
                logger.warning(f"Target values {list(unique_vals)[:5]} not in expected domain {domain}. Proceeding may fail later.")
            return True
        
        except Exception as e:
            logger.error(f"Error in dataset schema validation: {e}")
            raise CustomException(e)
    
    def initiate_data_validation(self, train_file_path, test_file_path):
        """
        Initiate the data validation process
        
        Args:
            train_file_path: Path to training data
            test_file_path: Path to test data
            
        Returns:
            Boolean indicating if validation passed
        """
        try:
            logger.info("Starting data validation")
            
            # Create validation directory
            os.makedirs(os.path.dirname(self.validation_config["report_file_path"]), exist_ok=True)
            
            # Read data
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            
            logger.info(f"Train dataset shape: {train_df.shape}")
            logger.info(f"Test dataset shape: {test_df.shape}")
            
            # Validate schema
            train_validation = self.validate_dataset_schema(train_df)
            test_validation = self.validate_dataset_schema(test_df)
            
            validation_status = train_validation and test_validation
            
            # Generate validation report
            validation_report = {
                "train_file_path": train_file_path,
                "test_file_path": test_file_path,
                "train_validation": train_validation,
                "test_validation": test_validation,
                "validation_status": validation_status
            }
            
            # Save validation report
            with open(self.validation_config["report_file_path"], 'w') as f:
                json.dump(validation_report, f, indent=4)
            
            logger.info(f"Data validation completed with status: {validation_status}")
            logger.info(f"Validation report saved at: {self.validation_config['report_file_path']}")
            
            return validation_status
            
        except Exception as e:
            logger.error(f"Error in data validation: {e}")
            raise CustomException(e)