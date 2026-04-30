<div align="center">
  <img src="https://img.icons8.com/color/96/000000/plant-under-sun.png" alt="AgroSmart Logo" width="80"/>
  <h1>AgroSmart</h1>
  <p><b>AI-powered Smart Agriculture Platform</b></p>
  <p>Crop recommendation, disease detection, weather insights, and market intelligence in one beautiful app.</p>
</div>

---

## 🚀 Features

- **AI Crop Recommendation:** Get personalized crop suggestions based on soil and climate data.
- **Plant Disease Detection:** Upload leaf images and detect diseases using deep learning.
- **Weather Dashboard:** Real-time weather and 7-day forecasts for major Indian cities.
- **Market Prices:** View current crop prices and trends from top markets.
- **Beautiful UI:** Modern, responsive design for desktop and mobile.

---

## 🗂️ Project Structure

```
Sunflower Project/
├── backend/           # Flask backend (API endpoints)
├── frontend/          # HTML, CSS, JS (UI)
├── data_ml/           # ML models, notebooks, datasets
│   ├── models/        # Trained model files (.pth, .pkl, .json)
│   └── notebooks/     # Jupyter notebooks for training
├── crop_information.csv
├── crop_recommendation_dataset.csv
├── market_prices.csv
├── weather_forecast.csv
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

---

## ⚡ Quick Start

1. **Clone the repo:**

   ```bash
   git clone https://github.com/Tejeswar001/Sunflower-Project.git
   cd Sunflower-Project
   ```

2. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask backend:**

   ```bash
   cd backend
   python app.py
   ```

4. **Open `index.html` in your browser** (or use a local server for full functionality).

---

## 🧠 Machine Learning

- **Crop Recommendation Engine (CRE):** Trained with a Gradient Boosting Classifier (`scikit-learn`) on soil and climate features, achieving ~99% accuracy.
- **Static Disease Detection:** MobileNetV3 deep learning model for single-image disease classification.
- **Disease Progression Module (DPDM):** CNN-LSTM hybrid model (MobileNetV2 backbone + Stacked LSTM) designed to analyze a 5-day spatiotemporal sequence of leaf images to predict disease class, severity score, and progression velocity.
- **Class Labels:** See `data_ml/models/class_labels.json` for all supported diseases.

---

## 🌐 APIs Used

- **Open-Meteo API** ([open-meteo.com](https://open-meteo.com/)) — Free weather forecast API for real-time and 7-day weather data.
- **Nominatim Geocoding API** ([nominatim.org](https://nominatim.org/)) — Free geocoding service to convert city names to latitude/longitude.
- **Agmarknet (Govt. of India)** ([agmarknet.gov.in](https://agmarknet.gov.in/)) — Official crop price data (used for sample/extension, not direct API).

---

## 🌦️ Weather & Market Data

- **Weather:** Powered by Open-Meteo API and Nominatim geocoding.
- **Market Prices:** Sample data included; can be extended with real-time sources.

---

## 📁 Notebooks & Datasets

- Jupyter notebooks for model training and evaluation in `data_ml/notebooks/`
- Datasets and model files in `data_ml/models/`

---

## 🖥️ Frontend

- Responsive UI with modern design
- Crop advisor, weather dashboard, disease detection, and market analytics
- Easy navigation and mobile support

---

## 🛠️ Tech Stack

- Python, Flask
- scikit-learn, PyTorch, torchvision
- HTML, CSS, JavaScript
- Chart.js for analytics

---

## 📚 Documentation

- All endpoints and features are documented in code comments and this README.
- For ML details, see the Jupyter notebooks in `data_ml/notebooks/`.

---

## 🤝 Contributing

Pull requests and suggestions are welcome! Please open an issue for major changes.

---

## 📄 License

This project is licensed under the MIT License
