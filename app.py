from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import requests
import json
import os
import google.generativeai as genai
import overpass
from functools import lru_cache
import asyncio
from typing import Optional, List, Dict, Any
import concurrent.futures
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import Depends, Request

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    text: str

class ModelManager:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._gemini_model = None
        self._model_lock = asyncio.Lock()
        self._tokenizer_lock = asyncio.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    async def _load_bert(self):
        """Asynchronously load BERT model and tokenizer"""
        try:
            async with self._model_lock:
                if self._model is None:
                    from transformers import TFBertForSequenceClassification
                    model_path = 'model'
                    self._model = await asyncio.get_event_loop().run_in_executor(
                        self._executor,
                        lambda: TFBertForSequenceClassification.from_pretrained(model_path)
                    )
            
            async with self._tokenizer_lock:
                if self._tokenizer is None:
                    from transformers import BertTokenizer
                    tokenizer_path = 'tokenizer'
                    self._tokenizer = await asyncio.get_event_loop().run_in_executor(
                        self._executor,
                        lambda: BertTokenizer.from_pretrained(tokenizer_path)
                    )
            return True
        except Exception as e:
            print(f"Error in _load_bert: {e}")
            return False

    async def load_bert_with_timeout(self, timeout_seconds: int = 10):
        """Load BERT with timeout"""
        try:
            return await asyncio.wait_for(self._load_bert(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return False

    async def _get_bert_prediction(self, text: str) -> Optional[float]:
        """Get BERT prediction"""
        try:
            inputs = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                lambda: self._tokenizer(
                    text,
                    return_tensors="tf",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
            )
            
            logits = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                lambda: self._model(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                ).logits
            )
            
            probabilities = tf.nn.softmax(logits, axis=1).numpy()
            return float(probabilities[0][1])
        except Exception as e:
            print(f"Error in _get_bert_prediction: {e}")
            return None

    async def get_bert_prediction(self, text: str, timeout_seconds: int = 5) -> Optional[float]:
        """Get BERT prediction with timeout"""
        try:
            return await asyncio.wait_for(self._get_bert_prediction(text), timeout_seconds)
        except asyncio.TimeoutError:
            return None

    @property
    def gemini_model(self):
        if self._gemini_model is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set.")
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        return self._gemini_model

    async def cleanup(self):
        """Release resources"""
        async with self._model_lock:
            if self._model is not None:
                del self._model
                self._model = None
        async with self._tokenizer_lock:
            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None
        if self._gemini_model is not None:
            del self._gemini_model
            self._gemini_model = None
        self._executor.shutdown(wait=False)

model_manager = ModelManager()

@lru_cache(maxsize=100)
async def get_geolocation(ip_address: str) -> Optional[Dict[str, Any]]:
    try:
        abstract_api_key = os.environ.get("ABSTRACT_API_KEY")
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: requests.get(f"https://ipgeolocation.abstractapi.com/v1/?api_key={abstract_api_key}&ip_address={ip_address}")
        )
        print(response.json())
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Geolocation lookup error: {e}")
        return None

@lru_cache(maxsize=50)
async def get_support_centers_from_overpass(lat: float, lon: float, radius: int = 5000) -> List[Dict[str, Any]]:
    try:
        api = overpass.API()
        query = f'node["amenity"~"hospital|clinic|doctors|social_facility|community_centre"](around:{radius},{lat},{lon});'
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: api.get(query, responseformat="geojson")
        )
        
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

async def get_gemini_response(prompt: str) -> Optional[str]:
    """Asynchronously fetch response from Gemini API"""
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model_manager.gemini_model.generate_content(prompt)
        )
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None
    
def get_client_ip(request: Request):
    """Get client IP address from request headers"""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0]
    return request.client.host


app.mount("/templates", StaticFiles(directory="."), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

@app.post("/predict")
async def predict_suicide_risk(request: PredictionRequest, client_ip: str = Depends(get_client_ip)):
    try:
        input_text = request.text.strip()
        if not input_text:
            raise HTTPException(status_code=400, detail="Invalid input. Please provide non-empty text.")

        # Load BERT model with timeout
        bert_available = await model_manager.load_bert_with_timeout(timeout_seconds=10)
        bert_risk_score = None
        
        if bert_available:
            bert_risk_score = await model_manager.get_bert_prediction(input_text, timeout_seconds=5)

        # Get Gemini prediction
        gemini_response = await get_gemini_response(
            f"Analyze the following text for potential suicide ideation: '{input_text}'. "
            "Reply with only a float value, with a risk score between 0 and 1"
        )
        
        try:
            gemini_risk_score = float(gemini_response) if gemini_response else 0.0
        except (ValueError, TypeError):
            print(f"Error parsing Gemini response: {gemini_response}")
            gemini_risk_score = 0.0

        # Calculate combined risk score
        if bert_risk_score is not None:
            weight_gemini = 0.8
            weight_bert = 0.2
            combined_risk_score = (gemini_risk_score * weight_gemini) + (bert_risk_score * weight_bert)
        else:
            combined_risk_score = gemini_risk_score

        # Get geolocation data and support centers if risk is high
        support_centers = []
        print(client_ip)
        if combined_risk_score > 0.5:
            geo_data = await get_geolocation(client_ip)
            print(geo_data)
            if geo_data and 'latitude' in geo_data and 'longitude' in geo_data:
                support_centers = await get_support_centers_from_overpass(
                    float(geo_data['latitude']),
                    float(geo_data['longitude'])
                )

        return {
            "risk_score": combined_risk_score,
            "is_high_risk": combined_risk_score > 0.5,
            "bert_risk_score": bert_risk_score if bert_risk_score is not None else "unavailable",
            "gemini_risk_score": gemini_risk_score,
            "model_status": "hybrid" if bert_risk_score is not None else "gemini_only",
            "support_centers": support_centers
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.on_event("shutdown")
async def shutdown_event():
    await model_manager.cleanup()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)