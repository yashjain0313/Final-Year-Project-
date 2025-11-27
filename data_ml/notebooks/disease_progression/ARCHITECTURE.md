# Disease Progression Detection - Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DISEASE PROGRESSION SYSTEM                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              FARMER WORKFLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

Day 1: Farmer notices spots on tomato leaves
   ↓
   📸 Takes photo → Uploads to app
   ↓
Day 2: Spots seem to be spreading
   ↓
   📸 Takes photo → Uploads to app
   ↓
Day 3: Leaves starting to yellow
   ↓
   📸 Takes photo → Uploads to app
   ↓
   🔬 Clicks "Analyze Progression"
   ↓
   ⚡ Gets instant results:
      • Disease: Tomato Late Blight
      • Severity: 72% (Severe)
      • Progression: Rapidly Worsening
      • Action: Immediate treatment needed!

┌─────────────────────────────────────────────────────────────────────────────┐
│                          TECHNICAL ARCHITECTURE                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  INPUT: Sequence of 5 Images (224x224x3 each)                           │
│  [Day 1] [Day 2] [Day 3] [Day 4] [Day 5]                                │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  FEATURE EXTRACTION (TimeDistributed CNN)                                │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  For each image in sequence:                                 │       │
│  │                                                               │       │
│  │  Image → EfficientNetB3 (Pretrained on ImageNet)             │       │
│  │         ↓                                                     │       │
│  │  Feature Map (1536 channels)                                 │       │
│  │         ↓                                                     │       │
│  │  Global Average Pooling                                      │       │
│  │         ↓                                                     │       │
│  │  Dense(1024, ReLU) → Dropout(0.3)                            │       │
│  │         ↓                                                     │       │
│  │  Dense(512, ReLU) → Feature Vector                           │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  Output: 5 feature vectors (one per day)                                │
│  Shape: (5, 512)                                                         │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  TEMPORAL PROCESSING (LSTM Layers)                                       │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  LSTM(256, return_sequences=True)                            │       │
│  │    • Processes sequence: Day1 → Day2 → Day3 → Day4 → Day5    │       │
│  │    • Learns temporal patterns                                │       │
│  │    • Remembers important changes                             │       │
│  │         ↓                                                     │       │
│  │  Dropout(0.3)                                                │       │
│  │         ↓                                                     │       │
│  │  LSTM(128, return_sequences=False)                           │       │
│  │    • Aggregates temporal information                         │       │
│  │    • Outputs final representation                            │       │
│  │         ↓                                                     │       │
│  │  Dropout(0.3)                                                │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  Output: Temporal feature vector                                        │
│  Shape: (128,)                                                           │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  DENSE LAYERS (Feature Combination)                                      │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  Dense(256, ReLU) → Dropout(0.4)                             │       │
│  │         ↓                                                     │       │
│  │  Dense(128, ReLU) → Dropout(0.3)                             │       │
│  └──────────────────────────────────────────────────────────────┘       │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  MULTI-TASK OUTPUT HEADS                                                 │
│                                                                          │
│  ┌─────────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │ Disease             │  │ Severity         │  │ Progression      │   │
│  │ Classification      │  │ Score            │  │ Rate             │   │
│  │                     │  │                  │  │                  │   │
│  │ Dense(38, Softmax)  │  │ Dense(1, Sigmoid)│  │ Dense(1, Linear) │   │
│  │                     │  │                  │  │                  │   │
│  │ Output:             │  │ Output:          │  │ Output:          │   │
│  │ "Tomato Late Blight"│  │ 0.72 (Severe)    │  │ 0.18/day         │   │
│  │ Confidence: 94%     │  │                  │  │ (Rapidly ↑)      │   │
│  └─────────────────────┘  └──────────────────┘  └──────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA FLOW EXAMPLE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Input Sequence:
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│ Day 1   │ Day 2   │ Day 3   │ Day 4   │ Day 5   │
│ [Image] │ [Image] │ [Image] │ [Image] │ [Image] │
│ 🌿      │ 🌿🟤    │ 🌿🟤🟤  │ 🟤🟤🟤  │ 🟫🟫🟫  │
│ Healthy │ Spots   │ Spread  │ Severe  │ Critical│
└─────────┴─────────┴─────────┴─────────┴─────────┘
     ↓         ↓         ↓         ↓         ↓
   CNN       CNN       CNN       CNN       CNN
     ↓         ↓         ↓         ↓         ↓
  [512]     [512]     [512]     [512]     [512]
     └─────────┴─────────┴─────────┴─────────┘
                       ↓
                     LSTM
                       ↓
                    [128]
                       ↓
                  Dense Layers
                       ↓
         ┌─────────────┼─────────────┐
         ↓             ↓             ↓
    Disease       Severity      Progression
  Late Blight       0.72           0.18

┌─────────────────────────────────────────────────────────────────────────────┐
│                      SYNTHETIC DATA GENERATION                              │
└─────────────────────────────────────────────────────────────────────────────┘

Original Image (Diseased Tomato)
         ↓
   Apply Progressive Augmentation
         ↓
┌─────────────────────────────────────────────────────────────┐
│ Day 1 (intensity=0.0): Original                             │
│   • No changes                                              │
│   • Severity: 0.2                                           │
├─────────────────────────────────────────────────────────────┤
│ Day 2 (intensity=0.25): Mild degradation                    │
│   • Hue shift: -7.5%                                        │
│   • Saturation: +12.5%                                      │
│   • Brightness: -5%                                         │
│   • Severity: 0.35                                          │
├─────────────────────────────────────────────────────────────┤
│ Day 3 (intensity=0.5): Moderate degradation                 │
│   • Hue shift: -15%                                         │
│   • Saturation: +25%                                        │
│   • Add 5 lesions                                           │
│   • Brightness: -10%                                        │
│   • Severity: 0.5                                           │
├─────────────────────────────────────────────────────────────┤
│ Day 4 (intensity=0.75): Severe degradation                  │
│   • Hue shift: -22.5%                                       │
│   • Saturation: +37.5%                                      │
│   • Add 7-8 lesions                                         │
│   • Brightness: -15%                                        │
│   • Severity: 0.65                                          │
├─────────────────────────────────────────────────────────────┤
│ Day 5 (intensity=1.0): Critical degradation                 │
│   • Hue shift: -30%                                         │
│   • Saturation: +50%                                        │
│   • Add 10 lesions                                          │
│   • Brightness: -20%                                        │
│   • Severity: 0.8                                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRAINING PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

1. Load PlantVillage Dataset
   ↓
2. For each image:
   • Generate 5-day progression sequence
   • Estimate severity from class name
   • Calculate progression rate
   ↓
3. Split data: 70% train, 15% val, 15% test
   ↓
4. Train model with:
   • Adam optimizer (lr=0.0001)
   • Multi-task loss (classification + regression)
   • Callbacks: EarlyStopping, ReduceLR, ModelCheckpoint
   ↓
5. Evaluate on test set:
   • Disease classification accuracy
   • Severity MAE
   • Progression rate MAE
   ↓
6. Save best model

┌─────────────────────────────────────────────────────────────────────────────┐
│                          API WORKFLOW                                       │
└─────────────────────────────────────────────────────────────────────────────┘

Farmer's Phone App
      ↓
POST /api/disease/upload-day
{
  "user_id": "farmer123",
  "day": 1,
  "image": "base64..."
}
      ↓
Backend stores image in memory
      ↓
Farmer uploads Day 2, Day 3...
      ↓
POST /api/disease/analyze-progression
{
  "user_id": "farmer123"
}
      ↓
Backend:
1. Retrieves all images for user
2. Prepares sequence (5 images)
3. Runs model prediction
4. Generates recommendations
      ↓
Response:
{
  "disease": "Tomato Late Blight",
  "confidence": 0.94,
  "severity_score": 0.72,
  "severity_level": "Severe",
  "progression_rate": 0.18,
  "progression_status": "Rapidly Worsening",
  "severity_timeline": [0.36, 0.54, 0.72],
  "recommendation": {
    "urgency": "high",
    "actions": ["Immediate treatment"],
    "treatments": ["Apply fungicide"]
  }
}
      ↓
App displays results with charts

┌─────────────────────────────────────────────────────────────────────────────┐
│                          KEY INNOVATIONS                                    │
└─────────────────────────────────────────────────────────────────────────────┘

1. ✨ Temporal Analysis
   • Not just "what disease" but "how fast spreading"
   • LSTM captures progression patterns
   • Better accuracy than single-image models

2. ✨ Synthetic Data Generation
   • No need for expensive temporal datasets
   • Realistic progression simulation
   • Can be improved with real data later

3. ✨ Multi-Task Learning
   • One model, three outputs
   • Shared representations improve accuracy
   • More efficient than separate models

4. ✨ Actionable Insights
   • Severity score (0-1)
   • Progression rate (per day)
   • Treatment urgency (low/medium/high)
   • Specific recommendations

5. ✨ Interpretability
   • Grad-CAM shows what model sees
   • Builds trust with farmers
   • Helps validate predictions

┌─────────────────────────────────────────────────────────────────────────────┐
│                          PERFORMANCE METRICS                                │
└─────────────────────────────────────────────────────────────────────────────┘

Target Performance:
┌─────────────────────────┬──────────┬────────────┐
│ Metric                  │ Target   │ Acceptable │
├─────────────────────────┼──────────┼────────────┤
│ Disease Accuracy        │ >90%     │ >85%       │
│ Top-3 Accuracy          │ >98%     │ >95%       │
│ Severity MAE            │ <0.10    │ <0.15      │
│ Progression MAE         │ <0.05    │ <0.08      │
│ Inference Time          │ <2s      │ <3s        │
│ Model Size              │ <200MB   │ <300MB     │
└─────────────────────────┴──────────┴────────────┘

Training Time (50 epochs):
┌─────────────┬────────────┬──────────────┐
│ Hardware    │ Batch Size │ Time         │
├─────────────┼────────────┼──────────────┤
│ RTX 3080    │ 32         │ 2-3 hours    │
│ RTX 2070    │ 16         │ 3-4 hours    │
│ GTX 1060    │ 8          │ 6-8 hours    │
│ CPU (i7)    │ 4          │ 18-24 hours  │
└─────────────┴────────────┴──────────────┘
```

---

## Visual Representation of Progression

```
Day 1: 🌿 Healthy-looking leaves, small spots
       Severity: 20% ████░░░░░░░░░░░░░░░░

Day 2: 🌿🟤 Spots growing, slight yellowing
       Severity: 35% ███████░░░░░░░░░░░░░

Day 3: 🌿🟤🟤 More yellowing, spots spreading
       Severity: 50% ██████████░░░░░░░░░░

Day 4: 🟤🟤🟤 Significant browning, wilting
       Severity: 70% ██████████████░░░░░░

Day 5: 🟫🟫🟫 Critical, heavy damage
       Severity: 85% █████████████████░░░

Progression Rate: 0.16/day → RAPIDLY WORSENING ⚠️
Action Required: IMMEDIATE TREATMENT 🚨
```

---

This architecture diagram shows the complete flow from farmer's phone to AI prediction!
