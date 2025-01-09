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

app = Flask(__name__)
CORS(app)

class ModelManager:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._gemini_model = None
    
    @property
    def model(self):
        if self._model is None:
            from transformers import TFBertForSequenceClassification
            try:
                model_path = 'model'
                self._model = TFBertForSequenceClassification.from_pretrained(model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
        return self._model
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import BertTokenizer
            try:
                tokenizer_path = 'tokenizer'
                self._tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            except Exception as e:
                print(f"Error loading tokenizer: {e}")
                raise
        return self._tokenizer
    
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
        if self._model is not None:
            del self._model
            self._model = None
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
    """
    Fetches support centers near the given location using the Overpass API.
    Now accepts lat/lon directly for better caching.
    """
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

        # Lazy load and use models
        inputs = model_manager.tokenizer(
            input_text, 
            return_tensors="tf", 
            padding=True, 
            truncation=True, 
            max_length=512
        )

        logits = model_manager.model(
            inputs["input_ids"], 
            attention_mask=inputs["attention_mask"]
        ).logits
        probabilities = tf.nn.softmax(logits, axis=1).numpy()
        bert_risk_score = float(probabilities[0][1])

        gemini_response = get_gemini_response(
            f"Analyze the following text for potential suicide ideation: '{input_text}'. "
            "Reply with only a float value, with a risk score between 0 and 1"
        )
        
        try:
            gemini_risk_score = float(gemini_response)
        except (ValueError, TypeError):
            print(f"Error parsing Gemini response: {gemini_response}")
            gemini_risk_score = 0.0

        weight_gemini = 0.8
        weight_bert = 0.2
        combined_risk_score = (gemini_risk_score * weight_gemini) + (bert_risk_score * weight_bert)

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
            "bert_risk_score": bert_risk_score,
            "gemini_risk_score": gemini_risk_score,
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