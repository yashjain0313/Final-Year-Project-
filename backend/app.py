
import requests
# Main Flask app for AgroSmart

from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import numpy as np
import os
import requests
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import json

app = Flask(__name__)

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '../data_ml/models/crop_recommendation/crop_recommendation_model.pkl')
    encoder_path = os.path.join(os.path.dirname(__file__), '../data_ml/models/crop_recommendation/label_encoder.pkl')
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder

model, encoder = None, None
try:
    model, encoder = load_model()
except Exception:
    pass

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    if model is None or encoder is None:
        print('ERROR: Model or encoder not loaded')
        return jsonify({'error': 'Model or encoder not loaded'}), 500
    # Validate and print input
    def clamp(val, minv, maxv):
        try:
            v = float(val)
            return max(minv, min(maxv, v))
        except:
            return minv
    try:
        features = pd.DataFrame([{
            'N': clamp(data.get('N', 0), 0, 140),
            'P': clamp(data.get('P', 0), 5, 145),
            'K': clamp(data.get('K', 0), 5, 205),
            'temperature': clamp(data.get('temperature', 0), 8, 43),
            'humidity': clamp(data.get('humidity', 0), 14, 100),
            'ph': clamp(data.get('ph', 0), 3.5, 9.9),
            'rainfall': clamp(data.get('rainfall', 0), 20, 300)
        }])
        print('Received features:', features.to_dict(orient='records'))
        predicted_label = model.predict(features)[0]
        print('Predicted label:', predicted_label)
        crop_name = encoder.inverse_transform([predicted_label])[0]
        print('Crop name:', crop_name)

        # Reference crop info and market price (from frontend/app.js data)
        crop_info = {
            "rice": {
                "name": "Rice",
                "season": "Kharif/Rabi",
                "duration": "120-150 days",
                "yield": "40-60 quintals/hectare",
                "description": "Rice is the staple food for billions of people. It requires flooded fields and warm climate.",
                "tips": "Prepare nursery beds, Transplant after 25-30 days, Maintain water level, Apply fertilizers in splits"
            },
            "wheat": {
                "name": "Wheat",
                "season": "Rabi",
                "duration": "120-150 days",
                "yield": "35-50 quintals/hectare",
                "description": "Wheat is a major cereal grain and staple food crop grown in temperate regions.",
                "tips": "Sow in October-November, Use certified seeds, Apply irrigation at critical stages, Control weeds timely"
            },
            "cotton": {
                "name": "Cotton",
                "season": "Kharif",
                "duration": "180-200 days",
                "yield": "15-25 quintals/hectare",
                "description": "Cotton is the most important fiber crop providing raw material to textile industry.",
                "tips": "Deep ploughing, Use Bt cotton varieties, Integrated pest management, Pick cotton when bolls open"
            },
            "tomato": {
                "name": "Tomato",
                "season": "Kharif/Rabi",
                "duration": "90-120 days",
                "yield": "400-600 quintals/hectare",
                "description": "Tomato is a popular vegetable crop rich in vitamins and minerals.",
                "tips": "Use disease-resistant varieties, Provide support to plants, Regular watering, Harvest at proper maturity"
            },
            "potato": {
                "name": "Potato",
                "season": "Rabi",
                "duration": "70-120 days",
                "yield": "200-400 quintals/hectare",
                "description": "Potato is an important food crop and source of carbohydrates.",
                "tips": "Plant in well-drained soil, Earthing up is essential, Control late blight disease, Harvest when plants mature"
            }
        }
        market_prices = {
            "rice": {"price": 2443.26, "trend": "+5.2%", "market": "Delhi"},
            "wheat": {"price": 2167.25, "trend": "+2.1%", "market": "Mumbai"},
            "cotton": {"price": 5906.01, "trend": "-1.8%", "market": "Bangalore"},
            "tomato": {"price": 2000.50, "trend": "+8.5%", "market": "Chennai"},
            "potato": {"price": 1200.75, "trend": "-3.2%", "market": "Kolkata"}
        }

        # Compose response
        info = crop_info.get(crop_name.lower(), {})
        market = market_prices.get(crop_name.lower(), {})
        response = {
            "recommendation": crop_name,
            "name": info.get("name", crop_name),
            "season": info.get("season", "N/A"),
            "duration": info.get("duration", "N/A"),
            "yield": info.get("yield", "N/A"),
            "description": info.get("description", "No description available."),
            "tips": info.get("tips", ""),
            "marketPrice": market,
            "confidence": 100  # Model does not provide, so set 100
        }
        return jsonify(response)
    except Exception as e:
        import traceback
        print('Crop recommendation error:', traceback.format_exc())
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/')
def home():
    frontend_dir = os.path.join(os.path.dirname(__file__), '../frontend')
    return send_from_directory(frontend_dir, 'index.html')

@app.route('/style.css')
def style_css():
    frontend_dir = os.path.join(os.path.dirname(__file__), '../frontend')
    return send_from_directory(frontend_dir, 'style.css')

@app.route('/app.js')
def app_js():
    frontend_dir = os.path.join(os.path.dirname(__file__), '../frontend')
    return send_from_directory(frontend_dir, 'app.js')

@app.route('/style.css')
def style():
    return send_from_directory('..', 'style.css')

@app.route('/app.js')
def appjs():
    return send_from_directory('..', 'app.js')


@app.route('/api/weather', methods=['GET'])
def get_weather():
    city = request.args.get('city', 'Delhi')
    # Geocode city to lat/lon using Nominatim
    geo_headers = {
        "User-Agent": "SunflowerProject/1.0 (contact: your@email.com)"
    }
    geocode_url = f'https://nominatim.openstreetmap.org/search?format=json&q={city}'
    try:
        geo_resp = requests.get(geocode_url, headers=geo_headers, timeout=10)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        if not geo_data:
            return jsonify({'error': 'City not found'}), 404
        lat = geo_data[0]['lat']
        lon = geo_data[0]['lon']
        # Fetch weather forecast from Open-Meteo
        weather_url = (
            f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}'
            '&hourly=temperature_2m,relative_humidity_2m'
            '&daily=temperature_2m_max,temperature_2m_min,precipitation_sum'
            '&current_weather=true&timezone=auto'
        )
        weather_resp = requests.get(weather_url, timeout=10)
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()
        # Prepare frontend-friendly response
        # Add daily humidity if available
        daily = weather_data.get('daily', {})
        # Open-Meteo daily humidity is not default; try to get from hourly for today
        humidity = None
        rainfall = None
        try:
            # Get today's hourly humidity and average it
            if 'relative_humidity_2m' in weather_data.get('hourly', {}):
                today_humidities = weather_data['hourly']['relative_humidity_2m'][:24]
                humidity = round(sum(today_humidities) / len(today_humidities), 1)
            # Get today's rainfall from daily
            if 'precipitation_sum' in daily:
                rainfall = daily['precipitation_sum'][0]
        except Exception:
            humidity = None
            rainfall = None
        result = {
            'city': city,
            'latitude': lat,
            'longitude': lon,
            'current': weather_data.get('current_weather', {}),
            'daily': daily,
            'hourly': weather_data.get('hourly', {}),
            'humidity': humidity,
            'rainfall': rainfall,
        }
        print(result)
        return jsonify(result)
    except Exception as e:
        import traceback
        print('Weather API error:', traceback.format_exc())
        return jsonify({'error': f'Weather fetch failed: {str(e)}'}), 500

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json

# Load disease detection model and labels
def load_disease_model():
    model_path = os.path.join(os.path.dirname(__file__), '../data_ml/models/mobilenetv3_plant_disease.pth')
    labels_path = os.path.join(os.path.dirname(__file__), '../data_ml/models/class_labels.json')
    if not os.path.exists(model_path):
        print(f'ERROR: Disease model file not found at {model_path}')
        return None, None
    if not os.path.exists(labels_path):
        print(f'ERROR: Disease labels file not found at {labels_path}')
        return None, None
    try:
        with open(labels_path, 'r') as f:
            class_labels = json.load(f)
        model = models.mobilenet_v3_small(pretrained=False)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_labels))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    except Exception as e:
        print(f'ERROR loading disease model or labels: {e}')
        return None, None
    return model, class_labels

disease_model, disease_labels = None, None
try:
    disease_model, disease_labels = load_disease_model()
except Exception:
    pass

@app.route('/api/disease', methods=['POST'])
def detect_disease():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    try:
        if disease_model is None or disease_labels is None:
            print('ERROR: Disease model or labels not loaded')
            return jsonify({'error': 'Model or labels not loaded on server'}), 500
        img = Image.open(file.stream).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = disease_model(input_tensor)
            pred_idx = output.argmax(dim=1).item()
            label = disease_labels.get(str(pred_idx), 'Unknown Disease')
            description = disease_labels.get(f"desc_{pred_idx}", '')
        return jsonify({'label': label, 'description': description})
    except Exception as e:
        import traceback
        print('Exception in /api/disease:', traceback.format_exc())
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(debug=True)
