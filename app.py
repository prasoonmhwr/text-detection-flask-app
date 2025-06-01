import os
import logging
import time
from functools import wraps
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification,RobertaTokenizer, pipeline
from werkzeug.exceptions import BadRequest, InternalServerError
import redis
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('api_request_duration_seconds', 'API request latency')
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions made')

class Config:
    MODEL_NAME = os.getenv('MODEL_NAME', 'prasoonmhwr/ai_detection_model')
    MAX_LENGTH = int(os.getenv('MAX_LENGTH', '512'))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    RATE_LIMIT = os.getenv('RATE_LIMIT', '100 per hour')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'


app = Flask(__name__)
app.config.from_object(Config)

CORS(app)

# Rate limiting
try:
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        default_limits=[Config.RATE_LIMIT],
        storage_uri=Config.REDIS_URL
    )
except:
    logger.warning("Redis not available, rate limiting disabled")
    limiter = None

class ModelManager:
    """Manages model loading and inference"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the Hugging Face model and tokenizer"""
        try:
            logger.info(f"Loading model: {Config.MODEL_NAME}")
            start_time = time.time()
            
            # Load tokenizer and model
            self.tokenizer = RobertaTokenizer.from_pretrained(Config.MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_NAME)
            
            # Move model to appropriate device
            self.model.to(Config.DEVICE)
            self.model.eval()

            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if Config.DEVICE == 'cuda' else -1,
                return_all_scores=True
            )

            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds on {Config.DEVICE}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    # def predict(self, text: str) -> Dict[str, Any]:
    #     """Make prediction on input text"""
    #     if not self.pipeline:
    #         raise RuntimeError("Model not loaded")
        
    #     try:
    #         # Validate input
    #         if not text or len(text.strip()) == 0:
    #             raise ValueError("Empty text provided")
            
    #         if len(text) > Config.MAX_LENGTH * 4:  # Rough character limit
    #             raise ValueError(f"Text too long. Maximum length: {Config.MAX_LENGTH * 4} characters")
            
    #         # Make prediction
    #         start_time = time.time()
            
    #         results = self.pipeline(text)
    #         inference_time = time.time() - start_time
            
    #         # Format results
    #         predictions = []
    #         for result in results[0]:  # Pipeline returns list of lists
    #             predictions.append({
    #                 'label': result['label'],
    #                 'confidence': round(result['score'], 4)
    #             })
            
    #         # Sort by confidence
    #         predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
    #         return {
    #             'predictions': predictions,
    #             'inference_time': round(inference_time, 4),
    #             'model_name': Config.MODEL_NAME
    #         }
            
    #     except Exception as e:
    #         logger.error(f"Prediction error: {str(e)}")
    #         raise
    def predict(self, text: str) -> Dict[str, Any]:
        """Make prediction on input text for AI detection"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            # Validate input
            if not text or len(text.strip()) == 0:
                raise ValueError("Empty text provided")
            
            # For AI detection, use shorter length limit (model trained on 128 tokens)
            if len(text) > 1024:  # Rough character limit for 128 tokens
                raise ValueError(f"Text too long. Maximum length: 512 characters")
            
            # Make prediction
            start_time = time.time()
            
            # Tokenize the input text (following the HF example exactly)
            inputs = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=258,  
                padding='max_length',
                # truncation=True,
                return_tensors='pt'
            ).to(Config.DEVICE)
            
            # Get prediction (no gradient calculation needed)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                # Use sigmoid for binary classification (not softmax)
                ai_probability = torch.sigmoid(logits).cpu().numpy().flatten()[0]
            
            inference_time = time.time() - start_time
            
            # Calculate human probability (complement of AI probability)
            human_probability = 1.0 - ai_probability
            
            # Format results - create both labels with their probabilities
            predictions = [
                {
                    'label': 'HUMAN',
                    'confidence': round(float(human_probability), 4)
                },
                {
                    'label': 'AI', 
                    'confidence': round(float(ai_probability), 4)
                }
            ]
            
            # Sort by confidence (highest first)
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Add interpretation
            if ai_probability > 0.5:
                primary_prediction = "AI-generated"
                certainty = ai_probability
            else:
                primary_prediction = "Human-written"
                certainty = human_probability
                
            return {
                'predictions': predictions,
                'primary_prediction': primary_prediction,
                'certainty': round(float(certainty), 4),
                'ai_probability': round(float(ai_probability), 4),
                'human_probability': round(float(human_probability), 4),
                'inference_time': round(inference_time, 4),
                # 'model_name': Config.MODEL_NAME
            }
            
        except Exception as e:
            logger.error(f"AI detection prediction error: {str(e)}")
            raise
# Initialize model manager
model_manager = ModelManager()

def monitor_request(f):
    """Decorator to monitor API requests"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        try:
            response = f(*args, **kwargs)
            status = response[1] if isinstance(response, tuple) else 200
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.endpoint,
                status=status
            ).inc()
            return response
        except Exception as e:
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.endpoint,
                status=500
            ).inc()
            raise
        finally:
            REQUEST_LATENCY.observe(time.time() - start_time)
    
    return decorated_function

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Basic model check
        if model_manager.model is None:
            return jsonify({'status': 'unhealthy', 'reason': 'Model not loaded'}), 503
        
        return jsonify({
            'status': 'healthy',
            'model': Config.MODEL_NAME,
            'device': Config.DEVICE,
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'reason': str(e)}), 503


@app.route('/predict', methods=['POST'])
@monitor_request
def predict():
    if limiter:
        limiter.limit(Config.RATE_LIMIT)(lambda: None)()
    
    try:
        # Validate request
        if not request.is_json:
            raise BadRequest("Request must be JSON")
        
        data = request.get_json()
        if not data:
            raise BadRequest("No JSON data provided")
        
        text = data.get('text')
        if not text:
            raise BadRequest("'text' field is required")
        
        if not isinstance(text, str):
            raise BadRequest("'text' must be a string")
        
        # Make prediction
        result = model_manager.predict(text)
        PREDICTION_COUNT.inc()
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    except BadRequest as e:
        logger.warning(f"Bad request: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
        
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500
    
@app.route('/batch-predict', methods=['POST'])
@monitor_request
def batch_predict():
    """Batch prediction endpoint"""
    if limiter:
        limiter.limit("50 per hour")(lambda: None)()
    
    try:
        if not request.is_json:
            raise BadRequest("Request must be JSON")
        
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not isinstance(texts, list):
            raise BadRequest("'texts' must be a list")
        
        if len(texts) == 0:
            raise BadRequest("'texts' list cannot be empty")
        
        if len(texts) > Config.BATCH_SIZE:
            raise BadRequest(f"Batch size cannot exceed {Config.BATCH_SIZE}")
        
        # Process batch
        results = []
        for i, text in enumerate(texts):
            try:
                result = model_manager.predict(text)
                results.append({
                    'index': i,
                    'success': True,
                    'data': result
                })
            except Exception as e:
                results.append({
                    'index': i,
                    'success': False,
                    'error': str(e)
                })
        
        PREDICTION_COUNT.inc(len(texts))
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except BadRequest as e:
        print("____________",e)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        return jsonify({
            'model_name': Config.MODEL_NAME,
            'device': Config.DEVICE,
            'max_length': Config.MAX_LENGTH,
            'batch_size': Config.BATCH_SIZE
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Run the application
    port = int(os.getenv('PORT', 5001))
    host = os.getenv('HOST', '0.0.0.0')
    
    logger.info(f"Starting Flask app on {host}:{port}")
    logger.info(f"Using model: {Config.MODEL_NAME}")
    logger.info(f"Device: {Config.DEVICE}")
    
    app.run(
        host=host,
        port=port,
        debug=Config.DEBUG,
        threaded=True
    )