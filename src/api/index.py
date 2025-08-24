from flask import Flask, request, render_template, jsonify
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from pipeline.predict_pipeline import PredictionPipeline, CustomData
from utils.logger import get_logger
from utils.exception import CustomException

# Initialize Flask app
app = Flask(__name__, template_folder='../templates', static_folder='../static')
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
            # Handle form input for single prediction
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

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        # Create CustomData object
        custom_data = CustomData(**data)
        df = custom_data.get_data_as_dataframe()
        
        # Make prediction
        prediction_pipeline = PredictionPipeline()
        prediction = prediction_pipeline.predict(df)[0]
        
        result = "Good Wafer" if prediction == 1 else "Bad Wafer"
        
        return jsonify({
            "success": True,
            "prediction": result,
            "prediction_value": int(prediction)
        })
    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# For local development
if __name__ == "__main__":
    app.run(debug=True)