# 🚀 Run Commands - Quick Reference

## ⚡ EASIEST WAY (Recommended)

### **Run Everything with One Command:**
```bash
.\start.bat
```
This automatically:
- ✅ Checks Python installation
- ✅ Installs dependencies
- ✅ Starts Flask backend (which serves frontend)
- ✅ ML model loads automatically

**Then open:** http://localhost:5000

---

## 🔧 MANUAL COMMANDS

### **Option 1: Run Backend (Serves Frontend + ML Model)**

```bash
# Navigate to backend folder
cd backend

# Install dependencies (first time only)
pip install -r requirements.txt

# Run Flask server
python app.py
```

**Access:** http://localhost:5000
- ✅ Frontend automatically served
- ✅ ML model automatically loaded
- ✅ All APIs active

---

### **Option 2: Run Frontend Separately (Development)**

If you want to run frontend independently for testing:

```bash
# Navigate to frontend folder
cd frontend

# Start simple HTTP server
python -m http.server 8000
```

**Access:** http://localhost:8000

⚠️ **Note:** API calls won't work without backend running!

---

### **Option 3: Train/Test ML Model**

#### **To Train the Models:**

**1. Crop Recommendation Engine (CRE):**
```bash
# Navigate to model training folder
cd data_ml/notebooks/crop_recomendation

# Train Gradient Boosting model
python train_cre.py
```

**2. Disease Progression Detection Module (DPDM):**
```bash
# Navigate to disease progression folder
cd data_ml/notebooks/disease_progression

# Step 1: Generate synthetic sequence data
python synthetic_sequence_generator.py

# Step 2: Train CNN-LSTM model
python train_dpdm.py

# Step 3: Evaluate model
python evaluate_dpdm.py
```

#### **To Test CRE Model Directly:**
```bash
cd data_ml/notebooks/crop_recomendation
python predictor.py --N 90 --P 42 --K 43 --temperature 20.8 --humidity 82 --ph 6.5 --rainfall 202
```

Then enter values when prompted:
```
Nitrogen (N) [0 - 140]: 90
Phosphorus (P) [5 - 145]: 42
Potassium (K) [5 - 205]: 43
Temperature (°C) [8 - 43]: 20.8
Humidity (%) [14 - 100]: 82
pH value [3.5 - 9.9]: 6.5
Rainfall (mm) [20 - 300]: 202
```

---

## 📋 COMPLETE WORKFLOW

### **For Development:**

**Terminal 1 - Backend:**
```bash
cd backend
python app.py
```

**Browser:**
```
Open: http://localhost:5000
```

That's it! Frontend and ML model are included.

---

### **For Production:**

```bash
cd backend

# Install production server
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## 🎯 WHAT RUNS WHERE

| Component | Command | Port | URL |
|-----------|---------|------|-----|
| **Backend + Frontend + ML** | `python app.py` | 5000 | http://localhost:5000 |
| **Frontend Only** | `python -m http.server 8000` | 8000 | http://localhost:8000 |
| **ML Model Test** | `python predict.py` | N/A | Terminal only |

---

## 🔄 TYPICAL USAGE

### **Daily Development:**
```bash
# Just run this:
.\start.bat

# Or manually:
cd backend
python app.py
```

### **Testing CRE Model:**
```bash
cd data_ml/notebooks/crop_recomendation
python predictor.py --N 90 --P 42 --K 43 --ph 6.5 --temperature 20 --humidity 80 --rainfall 200
```

### **Retraining CRE Model:**
```bash
cd data_ml/notebooks/crop_recomendation
python train_cre.py
```

### **Retraining Disease Progression Model:**
```bash
cd data_ml/notebooks/disease_progression
python synthetic_sequence_generator.py
python train_dpdm.py
```

---

## 🛑 STOP COMMANDS

### **Stop Backend Server:**
Press `Ctrl + C` in the terminal

### **Stop Frontend Server:**
Press `Ctrl + C` in the terminal

---

## 🐛 TROUBLESHOOTING

### **Problem: Port 5000 already in use**
```bash
# Find process using port 5000
netstat -ano | findstr :5000

# Kill the process (replace PID with actual number)
taskkill /PID <PID> /F

# Or run on different port
cd backend
flask run --port 5001
```

### **Problem: Module not found**
```bash
cd backend
pip install -r requirements.txt
```

### **Problem: Model not loading**
Check that these files exist:
- `data_ml/models/crop_recommendation/gradient_boosting_crop.pkl`
- `data_ml/models/crop_recommendation/standard_scaler.pkl`
- `data_ml/models/crop_recommendation/label_encoder.pkl`
- `data_ml/notebooks/disease_progression/models/dpdm/cnn_lstm_disease_final.keras`

---

## 📦 INSTALLATION (First Time)

```bash
# 1. Install Python dependencies
cd backend
pip install -r requirements.txt

# 2. Verify ML models exist
# Check: data_ml/models/crop_recommendation/

# 3. Run the application
python app.py
```

---

## 🎨 ARCHITECTURE SUMMARY

```
┌─────────────────────────────────────────────────────┐
│  ONE COMMAND: python app.py (in backend/)           │
│                                                      │
│  ┌────────────────────────────────────────────┐    │
│  │  Flask Backend (Port 5000)                 │    │
│  │  ├── Serves Frontend (HTML/CSS/JS)         │    │
│  │  ├── Loads ML Model (on startup)           │    │
│  │  ├── API: /api/recommend                   │    │
│  │  ├── API: /api/weather                     │    │
│  │  └── API: /api/disease                     │    │
│  └────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

---

## ✅ QUICK START CHECKLIST

- [ ] Navigate to project root
- [ ] Run `.\start.bat` OR `cd backend && python app.py`
- [ ] Open http://localhost:5000
- [ ] Test crop recommendation feature
- [ ] Done! 🎉

---

## 📝 SUMMARY

### **To Run Everything:**
```bash
.\start.bat
```
**OR**
```bash
cd backend
python app.py
```

### **To Test CRE Model:**
```bash
cd data_ml/notebooks/crop_recomendation
python predictor.py --N 90 --P 42 --K 43 --ph 6.5
```

### **To Retrain Models:**
```bash
cd data_ml/notebooks/crop_recomendation
python train_cre.py

cd ../disease_progression
python train_dpdm.py
```

---

## 🎯 RECOMMENDED WORKFLOW

1. **Start Development:**
   ```bash
   .\start.bat
   ```

2. **Open Browser:**
   ```
   http://localhost:5000
   ```

3. **Make Changes:**
   - Edit frontend files → Refresh browser
   - Edit backend files → Restart server (Ctrl+C, then python app.py)

4. **Stop Server:**
   ```
   Ctrl + C
   ```

---

**That's all you need! 🚀**

The backend serves everything - frontend, ML model, and all APIs!
