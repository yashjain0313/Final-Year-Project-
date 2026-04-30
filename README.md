<div align="center">
  <img src="https://img.icons8.com/color/96/000000/plant-under-sun.png" alt="AgroSmart Logo" width="80"/>
  <h1>AgroSmart</h1>
  <p><b>AI-powered Smart Agriculture Platform</b></p>
  <p>Crop recommendation, disease detection, weather insights, and market intelligence in one beautiful app.</p>
</div>

---

## 🚀 Features

- **User Authentication:** Secure login/signup system with session management powered by PostgreSQL.
- **AI Voice Assistant:** Voice-activated agricultural LLM to answer your farming queries instantly.
- **AI Crop Recommendation:** Get personalized crop suggestions based on soil and climate data, powered by OpenRouter LLMs.
- **Plant Disease Detection:** Upload leaf images and detect diseases. Images are stored securely in AWS S3 and analyzed by our AI pipeline.
- **Disease Progression Tracker:** Track crop health over multiple days to measure spread rate and severity, with a fully persisted history.
- **Weather Dashboard:** Real-time weather and 7-day forecasts for major Indian cities.
- **Market Prices:** View current crop prices and trends from top markets.

---

## 🗂️ Project Structure

```text
Sunflower Project/
├── backend/           # Flask backend (API endpoints, SQLAlchemy models, S3 integration)
├── frontend/          # HTML, CSS, JS (UI)
├── data_ml/           # ML models, notebooks, datasets
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

---

## ⚡ Quick Start

1. **Clone the repo:**

   ```bash
   git clone https://github.com/yashjain0313/Final-Year-Project-.git
   cd Final-Year-Project-
   ```

2. **Configure Environment Variables (`.env`):**
   Create a `.env` file in the root directory and add:
   ```env
   DATABASE_URL=postgresql://user:pass@host/db
   OPENROUTER_API_KEY=sk-or-v1-...
   AWS_ACCESS_KEY_ID=...
   AWS_SECRET_ACCESS_KEY=...
   AWS_REGION=ap-south-1
   S3_BUCKET_NAME=your-bucket-name
   ```

3. **Install Python dependencies:**

   ```bash
   pip install -r backend/requirements.txt
   ```

4. **Run the Flask backend:**

   ```bash
   python backend/app.py
   ```

5. **Open `http://127.0.0.1:5000` in your browser** to use the application.

---

## 🧠 AI & Machine Learning Infrastructure

- **OpenRouter LLMs:** Serves as the primary intelligence engine for the Voice Assistant, and acts as a dynamic fallback mechanism for Crop Recommendation and Disease Detection when local models are unavailable.
- **AWS S3 Cloud Storage:** Persists user-uploaded leaf imagery securely for the Instant Disease tool and the multi-day Progression Tracker.
- **Neon PostgreSQL:** Handles user accounts, passwords, and the `DiseaseHistory` records linking users to their S3 assets and AI diagnoses.

---

## 🌐 APIs Used

- **OpenRouter API** — Powering the Llama-3.3-70b-instruct conversational AI.
- **Open-Meteo API** — Free weather forecast API for real-time and 7-day weather data.
- **Nominatim Geocoding API** — Free geocoding service to convert city names to latitude/longitude.
- **AWS S3** — For robust, cloud-based blob storage of image assets.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, SQLAlchemy, Boto3 (AWS)
- **Database:** PostgreSQL (Neon DB)
- **AI Integration:** OpenRouter (Llama-3.3-70b)
- **Frontend:** Vanilla HTML5, CSS3, JavaScript (ES6+), Chart.js
- **ML Frameworks (Local):** scikit-learn, PyTorch, torchvision

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
