# 🌾 AgroSmart - Complete Setup Guide

## 📋 Overview
This guide will help you set up and run the crop recommendation ML model with proper frontend-backend integration.

## 🗂️ Project Structure
```
Final-Year-Project-/
├── backend/
│   ├── app.py                          # Flask backend server
│   ├── requirements.txt                # Python dependencies
│   ├── models/
│   ├── routes/
│   └── services/
├── frontend/
│   ├── index_new.html                  # Main HTML file
│   ├── app.js                          # Frontend JavaScript
│   └── style_new.css                   # Styles
├── data_ml/
│   ├── models/
│   │   ├── crop_recommendation/
│   │   │   ├── gradient_boosting_crop.pkl       # Gradient Boosting Model
│   │   │   ├── standard_scaler.pkl              # Feature Scaler
│   │   │   └── label_encoder.pkl                # Label Encoder
│   │   ├── mobilenetv3_plant_disease.pth        # Static Disease Detection Model
│   │   └── class_labels.json                    # Disease Labels
│   ├── notebooks/
│   │   └── disease_progression/
│   │       ├── models/dpdm/
│   │       │   └── cnn_lstm_disease_final.keras # Disease Progression Model
│   └── datasets/
└── crop_recommendation_dataset.csv
```

## 🚀 Step-by-Step Setup

### Step 1: Install Backend Dependencies

```powershell
# Navigate to backend folder
cd backend

# Install all required Python packages
pip install -r requirements.txt
```

**Required packages:**
- Flask (web framework)
- joblib (load ML models)
- pandas & numpy (data processing)
- scikit-learn (ML model)
- torch & torchvision (disease detection)
- Pillow (image processing)
- requests (API calls)

### Step 2: Verify ML Models Exist

Check that these files exist:
- ✅ `data_ml/models/crop_recommendation/gradient_boosting_crop.pkl`
- ✅ `data_ml/models/crop_recommendation/standard_scaler.pkl`
- ✅ `data_ml/models/crop_recommendation/label_encoder.pkl`
- ✅ `data_ml/models/mobilenetv3_plant_disease.pth`
- ✅ `data_ml/models/class_labels.json`
- ✅ `data_ml/notebooks/disease_progression/models/dpdm/cnn_lstm_disease_final.keras`

### Step 3: Run the Backend Server

```powershell
# From the backend folder
python app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

### Step 4: Access the Application

Open your browser and navigate to:
```
http://localhost:5000
```

The Flask backend automatically serves the frontend files!

## 🔄 How It Works

### Frontend → Backend Flow:

1. **User fills the form** (index_new.html)
   - Nitrogen (N)
   - Phosphorus (P)
   - Potassium (K)
   - pH level
   - Temperature
   - Humidity
   - Rainfall
   - State

2. **Frontend sends POST request** (app.js line 631)
   ```javascript
   fetch("/api/recommend", {
     method: "POST",
     headers: { "Content-Type": "application/json" },
     body: JSON.stringify(payload)
   })
   ```

3. **Backend receives request** (app.py line 33)
   - Validates input data
   - Clamps values to valid ranges
   - Creates feature DataFrame

4. **ML Model predicts** (app.py)
   ```python
   recs = cre_predictor.recommend_crop(...)
   crop_name = recs[0]['crop']
   confidence = recs[0]['confidence']
   ```

5. **Backend returns response** (app.py line 116-127)
   - Crop recommendation
   - Crop details (season, duration, yield)
   - Cultivation tips
   - Market price

6. **Frontend displays results** (app.js line 643-653)
   - Shows recommendation card
   - Displays crop information
   - Shows market price

## 🧪 Testing the System

### Test 1: Basic Crop Recommendation

**Input values:**
- N: 90
- P: 42
- K: 43
- pH: 6.5
- Temperature: 20.8
- Humidity: 82
- Rainfall: 202
- State: Punjab

**Expected Output:** Rice

### Test 2: Wheat Recommendation

**Input values:**
- N: 50
- P: 40
- K: 35
- pH: 6.5
- Temperature: 18
- Humidity: 65
- Rainfall: 50
- State: Haryana

**Expected Output:** Wheat

### Test 3: Cotton Recommendation

**Input values:**
- N: 120
- P: 60
- K: 50
- pH: 7.0
- Temperature: 28
- Humidity: 70
- Rainfall: 80
- State: Gujarat

**Expected Output:** Cotton

## 🐛 Troubleshooting

### Issue 1: Model Not Loading
**Error:** `Model or encoder not loaded`

**Solution:**
1. Check file paths in `app.py` (lines 21-22)
2. Verify model files exist
3. Check file permissions

### Issue 2: Import Errors
**Error:** `ModuleNotFoundError: No module named 'flask'`

**Solution:**
```powershell
pip install -r requirements.txt
```

### Issue 3: Port Already in Use
**Error:** `Address already in use`

**Solution:**
```powershell
# Find process using port 5000
netstat -ano | findstr :5000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or run on different port
flask run --port 5001
```

### Issue 4: CORS Errors
**Error:** `Access-Control-Allow-Origin`

**Solution:**
The backend already includes Flask-CORS in requirements.txt. If needed, add to `app.py`:
```python
from flask_cors import CORS
CORS(app)
```

### Issue 5: Frontend Not Loading
**Error:** 404 on CSS/JS files

**Solution:**
- Ensure you're accessing via `http://localhost:5000` (not opening HTML directly)
- Check that frontend files are in the correct location
- Verify Flask routes in `app.py` (lines 134-157)

## 📊 API Endpoints

### 1. Crop Recommendation
**Endpoint:** `POST /api/recommend`

**Request:**
```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "ph": 6.5,
  "temperature": 20.8,
  "humidity": 82,
  "rainfall": 202
}
```

**Response:**
```json
{
  "recommendation": "rice",
  "name": "Rice",
  "season": "Kharif/Rabi",
  "duration": "120-150 days",
  "yield": "40-60 quintals/hectare",
  "description": "Rice is the staple food...",
  "tips": "Prepare nursery beds...",
  "marketPrice": {
    "price": 2443.26,
    "trend": "+5.2%",
    "market": "Delhi"
  },
  "confidence": 99.85
}
```

### 2. Weather Data
**Endpoint:** `GET /api/weather?city=Delhi`

**Response:**
```json
{
  "city": "Delhi",
  "current": {...},
  "daily": {...},
  "humidity": 64.0,
  "rainfall": 21.3
}
```

### 3. Disease Detection
**Endpoint:** `POST /api/disease`

**Request:** Form-data with image file

**Response:**
```json
{
  "label": "Tomato - Early Blight",
  "description": "Early blight is a common disease..."
}
```

## 🎯 Features

### ✅ Implemented
- ✅ Crop recommendation using ML model
- ✅ Real-time weather data integration
- ✅ Disease detection using CNN
- ✅ Market price information
- ✅ Crop library with detailed information
- ✅ Location-based auto-fill
- ✅ Responsive design

### 🔄 Data Flow
1. User inputs soil and climate parameters
2. Frontend validates and sends to backend
3. Backend loads ML model (Gradient Boosting Classifier)
4. Model predicts best crop and calculates confidence
5. Backend enriches with crop info and market data
6. Frontend displays beautiful recommendation card

## 💡 Tips for Development

1. **Keep backend running** while testing frontend changes
2. **Check browser console** for JavaScript errors
3. **Check terminal** for backend errors
4. **Use browser DevTools Network tab** to inspect API calls
5. **Test with different input values** to verify model accuracy

## 🔐 Environment Variables (Optional)

Create `.env` file in backend folder:
```env
FLASK_ENV=development
FLASK_DEBUG=1
PORT=5000
```

## 📝 Quick Start Commands

```powershell
# Terminal 1: Run Backend
cd backend
pip install -r requirements.txt
python app.py

# Terminal 2: Open Browser
start http://localhost:5000
```

## 🎨 Frontend Features

- **Auto-fill from location**: Uses geolocation + weather API
- **Voice assistant**: Voice input for crop queries
- **Crop library**: Browse all available crops
- **Market prices**: Real-time market data
- **Weather forecasts**: 7-day weather charts
- **Disease detection**: Upload leaf images

## 🚦 Status Indicators

- ✅ **Green**: Model loaded successfully
- ⚠️ **Yellow**: Using cached data
- ❌ **Red**: Error loading model/data

## 📞 Support

If you encounter issues:
1. Check this guide first
2. Review error messages in terminal
3. Check browser console for frontend errors
4. Verify all dependencies are installed
5. Ensure ML model files exist

---

**Happy Farming! 🌾**
