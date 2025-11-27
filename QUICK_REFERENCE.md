# 🌾 Quick Reference - Crop Recommendation System

## ✅ System is Running!

**Backend Server:** http://localhost:5000
**Status:** ✅ Active

## 🎯 How to Use

### 1. Open Your Browser
Navigate to: **http://localhost:5000**

### 2. Fill the Crop Recommendation Form
Enter these parameters:
- **Nitrogen (N):** 0-140
- **Phosphorus (P):** 5-145  
- **Potassium (K):** 5-205
- **pH:** 3.5-9.9
- **Temperature:** 8-43°C
- **Humidity:** 14-100%
- **Rainfall:** 20-300mm
- **State:** Select from dropdown

### 3. Get Recommendations
Click "Get Recommendations" to receive:
- ✅ Best crop for your conditions
- ✅ Growing season & duration
- ✅ Expected yield
- ✅ Cultivation tips
- ✅ Current market price

## 📝 Sample Test Data

### Test 1: Rice
```
N: 90, P: 42, K: 43
pH: 6.5, Temp: 20.8°C
Humidity: 82%, Rainfall: 202mm
State: Punjab
```

### Test 2: Wheat
```
N: 50, P: 40, K: 35
pH: 6.5, Temp: 18°C
Humidity: 65%, Rainfall: 50mm
State: Haryana
```

### Test 3: Cotton
```
N: 120, P: 60, K: 50
pH: 7.0, Temp: 28°C
Humidity: 70%, Rainfall: 80mm
State: Gujarat
```

## 🔧 Other Features

### Weather Forecast
- Select a city from dropdown
- View 7-day forecast
- Get farming advisories

### Disease Detection
- Upload leaf image
- Get disease diagnosis
- Receive treatment recommendations

### Market Prices
- View current crop prices
- See price trends
- Analyze market conditions

### Crop Library
- Browse all crops
- Search by name
- View detailed information

## 🛑 To Stop the Server

Press `Ctrl + C` in the terminal

## 🔄 To Restart

Run: `.\start.bat`

## 📊 API Endpoints

- **POST** `/api/recommend` - Crop recommendation
- **GET** `/api/weather?city=Delhi` - Weather data
- **POST** `/api/disease` - Disease detection

## 🐛 Troubleshooting

**Problem:** Page not loading
**Solution:** Check if server is running on port 5000

**Problem:** Model error
**Solution:** Verify ML model files exist in `data_ml/models/`

**Problem:** Import errors
**Solution:** Run `pip install -r requirements.txt` in backend folder

---

**🎉 Enjoy using AgroSmart!**
