import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.utils.utils import save_object
from config.config import MODEL_TRAINER_CONFIG

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self):
        """
        Initialize model trainer with configuration
        """
        self.model_trainer_config = MODEL_TRAINER_CONFIG
    
    def train_model(self, X_train, y_train):
        """
        Train different models and return the best one
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Best model object
        """
        try:
            # Define models to train
            models = {
                "Random Forest": RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    class_weight="balanced",
                    random_state=42
                ),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs"
                ),
                "SVM": SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=False,
                    random_state=42
                ),
                "XGBoost": XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=42
                )
            }
            
            # Cross-validated evaluation for model selection
            model_report = {}
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            # Compute imbalance ratio for XGBoost
            try:
                pos = float(np.sum(y_train == 1))
                neg = float(np.sum(y_train == 0))
                scale_pos_weight = (neg / pos) if pos > 0 else 1.0
            except Exception:
                scale_pos_weight = 1.0
            for model_name, model in models.items():
                logger.info(f"Evaluating {model_name} with 5-fold CV")
                try:
                    # Set scale_pos_weight for XGBoost if available
                    if hasattr(model, "set_params") and model_name == "XGBoost":
                        model.set_params(scale_pos_weight=scale_pos_weight)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
                    cv_mean = float(np.mean(cv_scores))
                    model_report[model_name] = {
                        "model": model,
                        "cv_accuracy": cv_mean,
                        "cv_scores": cv_scores.tolist()
                    }
                    logger.info(f"{model_name} - CV Accuracy: {cv_mean:.4f}")
                except Exception as e:
                    logger.warning(f"{model_name} failed during CV: {e}")
            
            # Find best model based on CV accuracy
            best_model_name = max(model_report, key=lambda x: model_report[x]["cv_accuracy"])
            best_model = model_report[best_model_name]["model"]
            best_accuracy = model_report[best_model_name]["cv_accuracy"]
            
            logger.info(f"Best model: {best_model_name} with accuracy: {best_accuracy}")
            
            # Check if model meets base accuracy requirement
            if best_accuracy < self.model_trainer_config["base_accuracy"]:
                logger.warning(f"No model achieved the base accuracy of {self.model_trainer_config['base_accuracy']}. Proceeding with the best available model anyway.")
            
            # Fit the selected model on the full training data before returning
            best_model.fit(X_train, y_train)
            return best_model
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise CustomException(e)
    
    def initiate_model_training(self, train_array, test_array):
        """
        Initiate the model training process
        
        Args:
            train_array: Training data array
            test_array: Test data array
            
        Returns:
            Path to the trained model file
        """
        try:
            # Create model directory
            os.makedirs(os.path.dirname(self.model_trainer_config["model_file_path"]), exist_ok=True)
            
            logger.info("Splitting training and test data into features and target")
            
            # Split into features and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            # Train model
            model = self.train_model(X_train, y_train)
            
            # Evaluate model on test data
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred, zero_division=0)
            test_recall = recall_score(y_test, y_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_pred, zero_division=0)
            
            logger.info(f"Test metrics - Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1}")
            
            # Save model
            model_path = self.model_trainer_config["model_file_path"]
            save_object(model_path, model)
            
            logger.info(f"Model saved at: {model_path}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise CustomException(e)