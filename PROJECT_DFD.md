# AgroSmart Project - Data Flow Diagram (DFD)

## 📊 Visual DFD
*(Please refer to the generated `agrosmart_dfd.png` image for the visual representation)*

## 📝 Technical DFD (Mermaid)

Here is the structured data flow logic for the AgroSmart platform:

```mermaid
graph TD
    %% Entities
    User[👨‍🌾 Farmer]
    WeatherAPI[☁️ Weather API]
    MarketAPI[💰 Market API]
    
    %% Processes
    subgraph "AgroSmart System"
        UI[💻 Web Interface]
        Backend[⚙️ Flask Backend]
        
        subgraph "ML Inference Engine"
            P1(🌱 Crop Recommendation)
            P2(🔍 Disease Detection)
            P3(📈 Progression Analysis)
        end
    end
    
    %% Data Stores
    Models[(🧠 Trained Models)]
    
    %% Flows
    User -- "1. Input Soil/Location" --> UI
    User -- "2. Upload Leaf Photo" --> UI
    User -- "3. Upload 3-5 Day Photos" --> UI
    
    UI -- "JSON Request" --> Backend
    
    Backend -- "Features" --> P1
    Backend -- "Image Tensor" --> P2
    Backend -- "Image Sequence" --> P3
    
    P1 <--> Models
    P2 <--> Models
    P3 <--> Models
    
    Backend -- "Fetch Forecast" <--> WeatherAPI
    Backend -- "Fetch Prices" <--> MarketAPI
    
    P1 -- "Best Crop" --> Backend
    P2 -- "Disease Label" --> Backend
    P3 -- "Severity & Rate" --> Backend
    
    Backend -- "JSON Response" --> UI
    
    UI -- "4. Display Recommendations" --> User
    UI -- "5. Show Diagnosis & Treatment" --> User
    UI -- "6. Show Progression Chart" --> User
```

## 🔄 Process Descriptions

### 1. Crop Recommendation Flow
- **Input**: Nitrogen, Phosphorus, Potassium, pH, Rainfall, Temperature, Humidity.
- **Process**: Gradient Boosting Classifier analyzes soil parameters against trained patterns.
- **Output**: Top 5 recommended crops with confidence percentages.

### 2. Disease Detection Flow
- **Input**: Single leaf image.
- **Process**: MobileNetV3 CNN extracts features and classifies disease type.
- **Output**: Disease name, confidence score, and immediate treatment.

### 3. Disease Progression Flow (New)
- **Input**: Sequence of images (Day 1 to Day 5).
- **Process**: 
    - **CNN (EfficientNet)** extracts spatial features from each frame.
    - **LSTM** analyzes temporal changes across frames.
- **Output**: Severity score (%), Progression rate (speed), and urgency-based recommendations.
