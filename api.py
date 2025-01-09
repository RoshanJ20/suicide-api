import flask
from transformers import TFBertForSequenceClassification, BertTokenizer
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import requests
import json

app = Flask(__name__)
CORS(app)  

# Load the BERT model and tokenizer
def load_model_and_tokenizer():
    try:
        model_path = 'model'
        tokenizer_path = 'tokenizer'
        model = TFBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None, None

MODEL, TOKENIZER = load_model_and_tokenizer()

def get_geolocation(ip_address):
    try:
        response = requests.get(f'https://ipapi.co/{ip_address}/json/')
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Geolocation lookup error: {e}")
        return None

def fetch_local_support_centers(latitude, longitude):
    support_centers = [
        {
            "name": "Local Crisis Center",
            "phone": "1-800-HELP",
            "address": "123 Support St",
            "distance": "2 miles"
        }
    ]
    return support_centers

@app.route('/')
def serve_frontend():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_suicide_risk():
    try:
        # Get client IP (works behind proxy)
        ip_address = (
            request.headers.get('X-Forwarded-For', '').split(',')[0].strip() or  # From proxy
            request.headers.get('X-Real-IP') or  # From nginx
            request.remote_addr or  # Flask default
            '8.8.8.8'  # Google DNS IP as last resort
        )
        
        # Get input text from request
        input_text = request.json.get('text', '')
        
        # Preprocess text
        inputs = TOKENIZER(input_text, return_tensors="tf", padding=True, truncation=True, max_length=512)
        
        # Model prediction
        outputs = MODEL(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = outputs.logits
        
        # Convert logits to probabilities
        probabilities = tf.nn.softmax(logits, axis=1).numpy()
        
        # Assuming binary classification (suicide risk or no risk)
        risk_score = float(probabilities[0][1])  # probability of positive class
        
        # Geolocation lookup
        location_info = get_geolocation(ip_address)
        
        # Fetch local support centers if high risk
        support_centers = []
        if risk_score > 0.7:  # Adjust threshold as needed
            if location_info:
                support_centers = fetch_local_support_centers(
                    location_info.get('latitude'), 
                    location_info.get('longitude')
                )
        
        response = {
            "risk_score": risk_score,
            "is_high_risk": risk_score > 0.7,
            "location": {
                "city": location_info.get('city', 'Unknown') if location_info else 'Unknown',
                "region": location_info.get('region', 'Unknown') if location_info else 'Unknown',
                "country": location_info.get('country_name', 'Unknown') if location_info else 'Unknown'
            },
            "support_centers": support_centers
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

if __name__ == '__main__':
    app.run(debug=True)