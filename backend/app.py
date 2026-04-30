# Main Flask app for AgroSmart

import os
import sys
import json
import requests
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import uuid
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_folder='../frontend', static_url_path='')

# ── Database Configuration ─────────────────────────────────────
# PostgreSQL connection. User must set the DATABASE_URL environment variable.
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/agrosmart')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ── User Model ────────────────────────────────────────────────
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Ensure tables are created
with app.app_context():
    try:
        db.create_all()
        print("Database tables created/verified.")
    except Exception as e:
        print(f"Warning: Could not connect to the database. Make sure PostgreSQL is running and DATABASE_URL is configured. Error: {e}")

# ── Authentication Endpoints ──────────────────────────────────
@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.json or {}
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
        
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email is already registered'}), 400
        
    try:
        user = User(email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        return jsonify({'message': 'User created successfully', 'user_id': user.id, 'email': user.email})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json or {}
    email = data.get('email')
    password = data.get('password')
    
    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        return jsonify({'message': 'Login successful', 'user_id': user.id, 'email': user.email})
        
    return jsonify({'error': 'Invalid email or password'}), 401# ── Path helpers ──────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(__file__)
ML_BASE     = os.path.join(BACKEND_DIR, '../data_ml/notebooks')
sys.path.insert(0, os.path.join(ML_BASE, 'crop_recomendation'))
sys.path.insert(0, os.path.join(ML_BASE, 'disease_progression'))

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index_new.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# -- Module 1: Crop Recommendation Engine (CRE) ---------------
# Uses Gradient Boosting + real top-K confidence (report section 4.2)
cre_predictor = None
try:
    from predictor import CREPredictor
    cre_predictor = CREPredictor(
        model_path   = os.path.join(BACKEND_DIR, '../data_ml/models/crop_recommendation/gradient_boosting_crop.pkl'),
        scaler_path  = os.path.join(BACKEND_DIR, '../data_ml/models/crop_recommendation/standard_scaler.pkl'),
        encoder_path = os.path.join(BACKEND_DIR, '../data_ml/models/crop_recommendation/label_encoder.pkl'),
    )
    print('[OK] CRE (Gradient Boosting) loaded.')
except Exception as _cre_err:
    print('[WARN] CRE not loaded:', _cre_err)

# -- Module 3: Disease Progression Detection Module (DPDM) ----
# Uses CNN-LSTM (report section 4.4)
dpdm_predictor = None
try:
    from predict_dpdm import DPDMPredictor
    _dpdm_model = os.path.join(ML_BASE, 'disease_progression/models/dpdm/cnn_lstm_disease_final.keras')
    _dpdm_cls   = os.path.join(ML_BASE, 'disease_progression/models/dpdm/class_names.npy')
    if os.path.exists(_dpdm_model) and os.path.exists(_dpdm_cls):
        dpdm_predictor = DPDMPredictor(model_path=_dpdm_model, class_names_path=_dpdm_cls)
        print('[OK] DPDM (CNN-LSTM) loaded.')
    else:
        print('[WARN] DPDM model not found - run train_dpdm.py first.')
except Exception as _dpdm_err:
    print('[WARN] DPDM not loaded:', _dpdm_err)

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Crop recommendation using Gradient Boosting CRE (report §4.2)."""
    data = request.json
    if cre_predictor is None:
        return jsonify({'error': 'CRE model not loaded. Run train_cre.py first.'}), 503
    try:
        # Get top-5 recommendations with real confidence scores
        recs = cre_predictor.recommend_crop(
            nitrogen    = float(data.get('N', 56)),
            phosphorus  = float(data.get('P', 38)),
            potassium   = float(data.get('K', 27)),
            ph          = float(data.get('ph', 6.3)),
            temperature = float(data.get('temperature', 25)) if data.get('temperature') else None,
            humidity    = float(data.get('humidity', 70))    if data.get('humidity')    else None,
            rainfall    = float(data.get('rainfall', 100))   if data.get('rainfall')    else None,
            lat         = float(data['lat']) if data.get('lat') else None,
            lon         = float(data['lon']) if data.get('lon') else None,
            top_k       = 5,
        )
        crop_name  = recs[0]['crop']
        confidence = round(recs[0]['confidence'] * 100, 2)

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
    return send_from_directory(frontend_dir, 'index_new.html')

@app.route('/style.css')
@app.route('/style_new.css')
def style_css():
    frontend_dir = os.path.join(os.path.dirname(__file__), '../frontend')
    return send_from_directory(frontend_dir, 'style_new.css')

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

@app.route('/api/voice-chat', methods=['POST'])
def voice_chat():
    data = request.json or {}
    user_text = data.get('text')
    location = data.get('location', 'Unknown location')
    
    if not user_text:
        return jsonify({'error': 'No text provided'}), 400
        
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        api_key = os.environ.get('OPENROUTER_API_KEY', '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')
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = f"You are AgroSmart, an AI agricultural assistant. The user is located at {location}. Provide a brief, concise, and highly relevant farming or agricultural answer to the user's query. Keep it under 3 sentences."
        
        payload = {
            "model": "meta-llama/llama-3.3-70b-instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ]
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        
        result = response.json()
        reply_text = result['choices'][0]['message']['content']
        
        return jsonify({'response': reply_text})
        
    except Exception as e:
        import traceback
        print('Voice chat error:', traceback.format_exc())
        return jsonify({'error': f'Failed to process request: {str(e)}'}), 500

@app.route('/api/weather', methods=['GET'])
def get_weather():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    city = request.args.get('city', 'Unknown Location')
    
    try:
        # If lat/lon not provided, geocode the city name
        if not lat or not lon:
            geo_headers = {
                "User-Agent": "AgroSmartProject/1.0 (contact: test@example.com)"
            }
            geocode_url = f'https://nominatim.openstreetmap.org/search?format=json&q={city}'
            geo_resp = requests.get(geocode_url, headers=geo_headers, timeout=10)
            geo_resp.raise_for_status()
            geo_data = geo_resp.json()
            if not geo_data:
                return jsonify({'error': 'City not found'}), 404
            lat = geo_data[0]['lat']
            lon = geo_data[0]['lon']
            
        # Fetch weather forecast from Open-Meteo using exact coordinates
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


# ============================================
# DISEASE PROGRESSION DETECTION API (DPDM)
# CNN-LSTM — Report Section 4.4
# ============================================

# In-memory image store: user_id -> {day: np.ndarray}
_user_sequences = {}


@app.route('/api/disease-progression/upload-day', methods=['POST'])
def upload_day_image():
    """Store one day's leaf image for a user session."""
    import base64
    from io import BytesIO
    from datetime import datetime
    data    = request.json or {}
    user_id = data.get('user_id')
    day     = data.get('day')
    img_b64 = data.get('image')
    if not all([user_id, day is not None, img_b64]):
        return jsonify({'success': False, 'error': 'Missing user_id, day, or image'}), 400
    try:
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        img_arr = np.array(img)                        # uint8 (H,W,3)
        if user_id not in _user_sequences:
            _user_sequences[user_id] = {}
        _user_sequences[user_id][int(day)] = img_arr
        days = sorted(_user_sequences[user_id].keys())
        return jsonify({
            'success': True, 'user_id': user_id,
            'day': day, 'days_uploaded': days,
            'is_complete': len(days) >= 3,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/disease-progression/analyze', methods=['POST'])
def analyze_progression():
    """Run CNN-LSTM inference on the stored sequence (report §4.4)."""
    if dpdm_predictor is None:
        return jsonify({'success': False,
                        'error': 'DPDM model not loaded. Run train_dpdm.py first.'}), 503
    data    = request.json or {}
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'Missing user_id'}), 400
    images_dict = _user_sequences.get(user_id, {})
    if len(images_dict) < 3:
        return jsonify({'success': False,
                        'error': f'Need ≥3 images, got {len(images_dict)}'}), 400
    try:
        ordered = [images_dict[d] for d in sorted(images_dict.keys())]
        result  = dpdm_predictor.predict_from_images(ordered)
        del _user_sequences[user_id]     # clean up
        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/disease-progression/status', methods=['GET'])
def get_progression_status():
    """Return how many days have been uploaded for a user."""
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'Missing user_id'}), 400
    days = sorted(_user_sequences.get(user_id, {}).keys())
    return jsonify({
        'user_id': user_id, 'days_uploaded': days,
        'is_complete': len(days) >= 3,
    })


if __name__ == "__main__":
    app.run(debug=True)
