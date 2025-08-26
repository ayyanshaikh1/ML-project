import os
import pandas as pd
from flask import Flask, request, render_template, jsonify
from src.pipeline.predict_pipeline import PredictionPipeline, CustomData
from src.utils.logger import get_logger
from src.utils.exception import CustomException

app = Flask(__name__)
logger = get_logger(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    else:
        try:
            # Handle file upload
            if 'file' in request.files:
                file = request.files['file']
                if file.filename == '':
                    return render_template('predict.html', error="No file selected")
                
                # Check file extension
                if file.filename.endswith('.csv'):
                    data = pd.read_csv(file)
                elif file.filename.endswith(('.xls', '.xlsx')):
                    data = pd.read_excel(file)
                else:
                    return render_template('predict.html', error="Unsupported file format. Please upload CSV or Excel file.")
                
                logger.info(f"File uploaded: {file.filename}, Shape: {data.shape}")
                
                # Make predictions
                prediction_pipeline = PredictionPipeline()
                predictions = prediction_pipeline.predict(data)

                # Metrics
                good_count = int((predictions == 1).sum())
                bad_count = int((predictions == 0).sum())
                total_count = int(len(predictions))

                accuracy = None
                incorrect_count = None
                if 'Good/Bad' in data.columns:
                    y_true_series = data['Good/Bad']
                    if y_true_series.dtype == object:
                        mapping = {"Good": 1, "Bad": 0, "good": 1, "bad": 0, "GOOD": 1, "BAD": 0}
                        y_true_series = y_true_series.map(mapping)
                    try:
                        y_true = y_true_series.astype(int).values
                        accuracy = float((predictions == y_true).mean()) if total_count > 0 else None
                        incorrect_count = int((predictions != y_true).sum())
                    except Exception:
                        accuracy = None
                        incorrect_count = None

                # Add predictions to data for preview
                data_preview = data.copy()
                data_preview['Prediction'] = predictions
                data_preview['Prediction'] = data_preview['Prediction'].map({0: 'Bad Wafer', 1: 'Good Wafer'})

                # Convert to HTML table
                result_table = data_preview.head(100).to_html(classes='table table-striped')

                return render_template('predict.html', result_table=result_table,
                                      good_count=good_count,
                                      bad_count=bad_count,
                                      total_count=total_count,
                                      accuracy=accuracy,
                                      incorrect_count=incorrect_count)
            
            # Handle form input for single prediction
            else:
                # Get form data
                form_data = request.form.to_dict()
                
                # Create CustomData object
                custom_data = CustomData(**form_data)
                df = custom_data.get_data_as_dataframe()
                
                # Make prediction
                prediction_pipeline = PredictionPipeline()
                prediction = prediction_pipeline.predict(df)[0]
                
                result = "Good Wafer" if prediction == 1 else "Bad Wafer"
                
                return render_template('predict.html', single_prediction=result)
                
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return render_template('predict.html', error=str(e))

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'GET':
        return render_template('train.html')
    else:
        try:
            # Import here to avoid circular import
            from src.pipeline.train_pipeline import TrainPipeline
            
            # Handle file upload
            if 'file' not in request.files:
                return render_template('train.html', error="No file uploaded")
            
            file = request.files['file']
            if file.filename == '':
                return render_template('train.html', error="No file selected")
            
            # Save file temporarily
            temp_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", file.filename)
            file.save(temp_file_path)
            
            logger.info(f"File uploaded for training: {file.filename}")
            
            # Start training
            train_pipeline = TrainPipeline()
            model_path = train_pipeline.start_training(temp_file_path)
            
            return render_template('train.html', success=f"Model trained successfully and saved at: {model_path}")
            
        except Exception as e:
            logger.error(f"Error in training: {e}")
            return render_template('train.html', error=str(e))

if __name__ == "__main__":
    app.run(debug=True)