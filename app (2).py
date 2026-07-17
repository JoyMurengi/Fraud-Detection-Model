"""
Fraud Detection Model API

A production-ready Flask application for serving fraud detection predictions.
Provides REST API endpoints for real-time transaction scoring.

Author: ML Engineering Team
Date: November 2025
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

# Add the model path to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models', 'final_model', 'inference'))

try:
    from predict import FraudDetector
except ImportError as e:
    print(f"Error importing FraudDetector: {e}")
    sys.exit(1)

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for web applications

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables
fraud_detector = None
model_load_time = None

def load_model():
    """Load the fraud detection model on startup."""
    global fraud_detector, model_load_time
    try:
        start_time = datetime.now()
        model_dir = os.path.join(os.path.dirname(__file__), 'models', 'final_model')
        fraud_detector = FraudDetector(model_dir=model_dir)
        model_load_time = datetime.now()

        load_duration = (model_load_time - start_time).total_seconds()
        logger.info(f"Model loaded successfully in {load_duration:.2f} seconds")
        return True

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        if fraud_detector is None:
            return jsonify({
                'status': 'unhealthy',
                'message': 'Model not loaded',
                'timestamp': datetime.now().isoformat()
            }), 503

        return jsonify({
            'status': 'healthy',
            'message': 'Fraud detection API is running',
            'model_loaded': fraud_detector is not None,
            'load_time': model_load_time.isoformat() if model_load_time else None,
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint."""
    try:
        # Check if model is loaded
        if fraud_detector is None:
            return jsonify({
                'error': 'Model not loaded',
                'timestamp': datetime.now().isoformat()
            }), 503

        # Get request data
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'timestamp': datetime.now().isoformat()
            }), 400

        # Handle single transaction or batch
        if isinstance(data, dict):
            # Single transaction
            if 'transaction' in data:
                transaction_data = data['transaction']
            else:
                transaction_data = data

            threshold = data.get('threshold', 0.5)

        elif isinstance(data, list):
            # Batch transactions
            transaction_data = pd.DataFrame(data)
            threshold = 0.5

        else:
            return jsonify({
                'error': 'Invalid data format. Expected dict or list.',
                'timestamp': datetime.now().isoformat()
            }), 400

        # Make prediction
        start_time = datetime.now()
        result = fraud_detector.predict(transaction_data, threshold=threshold)
        prediction_time = (datetime.now() - start_time).total_seconds()

        # Prepare response
        response = {
            'prediction': result,
            'model_version': result.get('model_version', 'unknown'),
            'prediction_time_seconds': prediction_time,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Prediction completed in {prediction_time:.4f}s")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint for multiple transactions."""
    try:
        if fraud_detector is None:
            return jsonify({
                'error': 'Model not loaded',
                'timestamp': datetime.now().isoformat()
            }), 503

        data = request.get_json()

        if not data or 'transactions' not in data:
            return jsonify({
                'error': 'Expected JSON with "transactions" array',
                'timestamp': datetime.now().isoformat()
            }), 400

        transactions = data['transactions']
        threshold = data.get('threshold', 0.5)

        if not isinstance(transactions, list):
            return jsonify({
                'error': 'Transactions must be a list',
                'timestamp': datetime.now().isoformat()
            }), 400

        # Convert to DataFrame and predict
        df = pd.DataFrame(transactions)
        start_time = datetime.now()
        result = fraud_detector.predict(df, threshold=threshold)
        prediction_time = (datetime.now() - start_time).total_seconds()

        response = {
            'batch_prediction': result,
            'batch_size': len(transactions),
            'prediction_time_seconds': prediction_time,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Batch prediction ({len(transactions)} transactions) completed in {prediction_time:.4f}s")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information and metadata."""
    try:
        if fraud_detector is None:
            return jsonify({
                'error': 'Model not loaded',
                'timestamp': datetime.now().isoformat()
            }), 503

        info = fraud_detector.get_model_info()

        response = {
            'model_info': info,
            'api_version': '1.0',
            'load_time': model_load_time.isoformat() if model_load_time else None,
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Model info request failed: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/threshold', methods=['POST'])
def update_threshold():
    """Update the model decision threshold."""
    try:
        if fraud_detector is None:
            return jsonify({
                'error': 'Model not loaded',
                'timestamp': datetime.now().isoformat()
            }), 503

        data = request.get_json()

        if not data or 'threshold' not in data:
            return jsonify({
                'error': 'Expected JSON with "threshold" value',
                'timestamp': datetime.now().isoformat()
            }), 400

        threshold = data['threshold']

        if not isinstance(threshold, (int, float)) or not (0 < threshold < 1):
            return jsonify({
                'error': 'Threshold must be a number between 0 and 1',
                'timestamp': datetime.now().isoformat()
            }), 400

        fraud_detector.set_threshold(threshold)

        return jsonify({
            'message': f'Threshold updated to {threshold}',
            'new_threshold': threshold,
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Threshold update failed: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/', methods=['GET'])
def home():
    """API documentation endpoint."""
    documentation = {
        'service': 'Fraud Detection API',
        'version': '1.0',
        'endpoints': {
            'GET /': 'This documentation',
            'GET /health': 'Health check',
            'GET /model-info': 'Model metadata and performance',
            'POST /predict': 'Single transaction prediction',
            'POST /predict/batch': 'Batch transaction predictions', 
            'POST /threshold': 'Update decision threshold'
        },
        'usage': {
            'predict': {
                'method': 'POST',
                'url': '/predict',
                'body': {
                    'transaction': {
                        'TransactionAmt': 150.00,
                        'ProductCD': 'W',
                        '...': 'other features'
                    },
                    'threshold': 0.5
                }
            }
        },
        'timestamp': datetime.now().isoformat()
    }

    return jsonify(documentation), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'Please check the API documentation at GET /',
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'Please check server logs for details',
        'timestamp': datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    # Load model on startup
    logger.info("Starting Fraud Detection API...")

    if load_model():
        logger.info("Model loaded successfully. Starting Flask server...")

        # Get configuration from environment
        host = os.getenv('HOST', '0.0.0.0')
        port = int(os.getenv('PORT', 5000))
        debug = os.getenv('FLASK_ENV') == 'development'

        app.run(host=host, port=port, debug=debug)
    else:
        logger.error("Failed to load model. Exiting...")
        sys.exit(1)
