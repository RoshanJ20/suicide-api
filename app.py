import flask
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import requests
import json
import os
import google.generativeai as genai
import overpass
from functools import lru_cache
import threading
import concurrent.futures
from threading import Thread, Event

app = Flask(__name__)
CORS(app)

class ThreadWithResult(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
        self._error = None

    def run(self):
        if self._target is not None:
            try:
                self._return = self._target(*self._args, **self._kwargs)
            except Exception as e:
                self._error = e

    def join(self, timeout=None):
        Thread.join(self, timeout)
        if self._error is not None:
            raise self._error
        return self._return

def run_with_timeout(func, args=(), kwargs={}, timeout=10):
    """Run a function with timeout using Thread"""
    thread = ThreadWithResult(target=func, args=args, kwargs=kwargs)
    thread.daemon = True  # Daemon thread will be killed when main thread exits
    thread.start()
    try:
        return thread.join(timeout=timeout)
    except Exception as e:
        return None
    finally:
        # The thread will be terminated when the main thread exits
        # since it's a daemon thread
        pass

class ModelManager:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._gemini_model = None
        self._model_lock = threading.Lock()
        self._tokenizer_lock = threading.Lock()
    
    def _load_bert(self):
        """Internal method to load BERT model and tokenizer"""
        try:
            with self._model_lock:
                if self._model is None:
                    from transformers import TFBertForSequenceClassification
                    model_path = 'model'
                    self._model = TFBertForSequenceClassification.from_pretrained(model_path)
            
            with self._tokenizer_lock:
                if self._tokenizer is None:
                    from transformers import BertTokenizer
                    tokenizer_path = 'tokenizer'
                    self._tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            return True
        except Exception as e:
            print(f"Error in _load_bert: {e}")
            return False

    def load_bert_with_timeout(self, timeout_seconds=10):
        """Attempt to load BERT model with timeout"""
        return run_with_timeout(self._load_bert, timeout=timeout_seconds)

    def _get_bert_prediction(self, text):
        """Internal method for BERT prediction"""
        try:
            inputs = self._tokenizer(
                text,
                return_tensors="tf",
                padding=True,
                truncation=True,
                max_length=512
            )
            logits = self._model(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            ).logits
            probabilities = tf.nn.softmax(logits, axis=1).numpy()
            return float(probabilities[0][1])
        except Exception as e:
            print(f"Error in _get_bert_prediction: {e}")
            return None

    def get_bert_prediction(self, text, timeout_seconds=5):
        """Get BERT prediction with timeout"""
        return run_with_timeout(
            self._get_bert_prediction,
            args=(text,),
            timeout=timeout_seconds
        )
    
    @property
    def gemini_model(self):
        if self._gemini_model is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set.")
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        return self._gemini_model
    
    def cleanup(self):
        """Release resources when needed"""
        # Always acquire _model_lock first
        with self._model_lock:
            if self._model is not None:
                del self._model
                self._model = None
        # Then acquire _tokenizer_lock
        with self._tokenizer_lock:
            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None
        if self._gemini_model is not None:
            del self._gemini_model
            self._gemini_model = None
        
model_manager = ModelManager()

@lru_cache(maxsize=100)
def get_geolocation(ip_address):
    try:
        response = requests.get(f'https://ipapi.co/{ip_address}/json/')
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Geolocation lookup error: {e}")
        return None

@lru_cache(maxsize=50)
def get_support_centers_from_overpass(lat, lon, radius=5000):
    try:
        api = overpass.API()
        query = f'node["amenity"~"hospital|clinic|doctors|social_facility|community_centre"](around:{radius},{lat},{lon});'
        result = api.get(query, responseformat="geojson")
        
        support_centers = []
        for feature in result.get("features", []):
            properties = feature.get("properties", {})
            tags = properties.get("tags", {})
            geometry = feature.get("geometry", {})

            center = {
                "name": tags.get("name", "Unknown"),
                "address": tags.get("addr:street"),
                "phone": tags.get("phone"),
                "email": tags.get("email"),
                "website": tags.get("website"),
                "coordinates": geometry.get("coordinates")
            }
            support_centers.append(center)

        return support_centers

    except Exception as e:
        print(f"Error fetching support centers from Overpass: {e}")
        return []

def get_gemini_response(prompt):
    """Fetches a response from the Gemini API using the provided prompt."""
    try:
        response = model_manager.gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None

@app.route('/')
def serve_frontend():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_suicide_risk():
    try:
        input_text = request.json.get('text', '').strip()
        if not input_text:
            return jsonify({
                "error": "Invalid input. Please provide a non-empty text for prediction."
            }), 400

        # Try to load BERT model with timeout
        bert_available = model_manager.load_bert_with_timeout(timeout_seconds=10)
        bert_risk_score = None
        
        if bert_available:
            # Try to get BERT prediction with timeout
            bert_risk_score = model_manager.get_bert_prediction(input_text, timeout_seconds=5)

        # Get Gemini prediction
        gemini_response = get_gemini_response(
            f"Analyze the following text for potential suicide ideation: '{input_text}'. "
            "Reply with only a float value, with a risk score between 0 and 1"
        )
        
        try:
            gemini_risk_score = float(gemini_response)
        except (ValueError, TypeError):
            print(f"Error parsing Gemini response: {gemini_response}")
            gemini_risk_score = 0.0

        # Calculate combined risk score based on available models
        if bert_risk_score is not None:
            weight_gemini = 0.8
            weight_bert = 0.2
            combined_risk_score = (gemini_risk_score * weight_gemini) + (bert_risk_score * weight_bert)
        else:
            # Use only Gemini score if BERT is unavailable
            combined_risk_score = gemini_risk_score

        ip_address = request.headers.get('X-Forwarded-For', '').split(',')[0].strip() or request.remote_addr
        location_info = get_geolocation(ip_address)

        support_centers = []
        if combined_risk_score > 0.7 and location_info:
            support_centers = get_support_centers_from_overpass(
                location_info['latitude'],
                location_info['longitude']
            )

        response = {
            "risk_score": combined_risk_score,
            "is_high_risk": combined_risk_score > 0.5,
            "bert_risk_score": bert_risk_score if bert_risk_score is not None else "unavailable",
            "gemini_risk_score": gemini_risk_score,
            "model_status": "hybrid" if bert_risk_score is not None else "gemini_only",
            "support_centers": support_centers,
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "An error occurred during prediction"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

# Cleanup handler
@app.teardown_appcontext
def cleanup_resources(exception=None):
    model_manager.cleanup()

if __name__ == '__main__':
    app.run(debug=True)