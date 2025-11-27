# Disease Progression Detection - Setup Guide

## 📥 Dataset Download Instructions

### Option 1: PlantVillage Dataset (Recommended)

1. **Install Kaggle API**:
   ```bash
   pip install kaggle
   ```

2. **Setup Kaggle Credentials**:
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - Download `kaggle.json`
   - Place it in:
     - Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
     - Linux/Mac: `~/.kaggle/kaggle.json`

3. **Download Dataset**:
   ```bash
   # Navigate to datasets directory
   cd data_ml/datasets
   
   # Download PlantVillage dataset
   kaggle datasets download -d emmarex/plantdisease
   
   # Unzip
   unzip plantdisease.zip -d plant_disease
   ```

### Option 2: New Plant Diseases Dataset

```bash
cd data_ml/datasets
kaggle datasets download -d vipoooool/new-plant-diseases-dataset
unzip new-plant-diseases-dataset.zip -d plant_disease
```

### Option 3: Manual Download

1. Visit: https://www.kaggle.com/datasets/emmarex/plantdisease
2. Click "Download" button
3. Extract to `data_ml/datasets/plant_disease/`

---

## 📁 Expected Directory Structure

After download, your structure should look like:

```
Final-Year-Project-/
├── data_ml/
│   ├── datasets/
│   │   └── plant_disease/
│   │       ├── Pepper__bell___Bacterial_spot/
│   │       ├── Pepper__bell___healthy/
│   │       ├── Potato___Early_blight/
│   │       ├── Potato___Late_blight/
│   │       ├── Potato___healthy/
│   │       ├── Tomato___Bacterial_spot/
│   │       ├── Tomato___Early_blight/
│   │       ├── Tomato___Late_blight/
│   │       ├── Tomato___Leaf_Mold/
│   │       ├── Tomato___Septoria_leaf_spot/
│   │       ├── Tomato___Spider_mites/
│   │       ├── Tomato___Target_Spot/
│   │       ├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
│   │       ├── Tomato___Tomato_mosaic_virus/
│   │       ├── Tomato___healthy/
│   │       └── ... (38 classes total)
│   ├── notebooks/
│   │   └── disease_progression/
│   │       ├── disease_progression_model.py
│   │       ├── train_model.py
│   │       └── IMPLEMENTATION_PLAN.md
│   └── models/
│       └── (trained models will be saved here)
└── backend/
    └── disease_progression_api.py
```

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install tensorflow keras opencv-python numpy pandas matplotlib seaborn scikit-learn pillow albumentations
```

### Step 2: Download Dataset

Follow the instructions above to download the PlantVillage dataset.

### Step 3: Train the Model

```bash
cd data_ml/notebooks/disease_progression
python train_model.py
```

**Note**: Update the `DATASET_PATH` in `train_model.py` to match your dataset location:
```python
DATASET_PATH = '../../datasets/plant_disease'
```

### Step 4: Monitor Training

Training will create:
- `models/disease_progression_best.h5` - Best model checkpoint
- `models/disease_progression_final.h5` - Final trained model
- `logs/` - TensorBoard logs
- `training_history.png` - Training curves
- `confusion_matrix.png` - Confusion matrix
- `regression_plots.png` - Severity/progression plots

View TensorBoard:
```bash
tensorboard --logdir=logs/
```

### Step 5: Test the API

```bash
cd ../../../backend
python disease_progression_api.py
```

---

## 📊 Dataset Information

### PlantVillage Dataset

- **Total Images**: ~54,000
- **Number of Classes**: 38
- **Image Size**: 256×256 (will be resized to 224×224)
- **Format**: RGB JPEG
- **Crops Covered**: 
  - Tomato (10 classes)
  - Potato (3 classes)
  - Pepper (2 classes)
  - Apple (4 classes)
  - Cherry (2 classes)
  - Corn (4 classes)
  - Grape (4 classes)
  - Peach (2 classes)
  - Strawberry (2 classes)
  - And more...

### Class Distribution

The dataset is relatively balanced, but some classes have more images than others. The training script handles this with:
- Stratified splitting
- Class weights (optional)
- Data augmentation

---

## 🎯 Training Configuration

### Default Hyperparameters

```python
IMAGE_SIZE = (224, 224)
SEQUENCE_LENGTH = 5
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001
CNN_BACKBONE = 'efficientnet'  # or 'resnet50'
LSTM_UNITS = [256, 128]
DROPOUT_RATE = 0.3
```

### Modify Training

Edit `train_model.py` to change:
- Batch size (if GPU memory issues)
- Number of epochs
- Learning rate
- Model architecture

### GPU Recommendations

- **Minimum**: 6GB VRAM (NVIDIA GTX 1060 or better)
- **Recommended**: 8GB+ VRAM (RTX 2070 or better)
- **Training Time**: 
  - With GPU: 2-4 hours
  - Without GPU: 12-24 hours

### CPU Training

If you don't have a GPU:
1. Reduce batch size to 8 or 4
2. Use fewer LSTM units: `[128, 64]`
3. Consider using ResNet50 instead of EfficientNet (faster)

---

## 🧪 Testing the Model

### Test Single Sequence

```python
from tensorflow import keras
import numpy as np

# Load model
model = keras.models.load_model('models/disease_progression_final.h5')
class_names = np.load('models/class_names.npy', allow_pickle=True)

# Prepare test sequence (5 images, 224x224x3)
test_sequence = np.random.rand(1, 5, 224, 224, 3)

# Predict
predictions = model.predict(test_sequence)

disease_idx = np.argmax(predictions['disease_classification'][0])
print(f"Disease: {class_names[disease_idx]}")
print(f"Severity: {predictions['severity_score'][0][0]:.3f}")
print(f"Progression Rate: {predictions['progression_rate'][0][0]:.3f}")
```

---

## 📱 API Usage Examples

### Upload Day 1 Image

```bash
curl -X POST http://localhost:5000/api/disease/upload-day \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "farmer123",
    "day": 1,
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
  }'
```

### Check Status

```bash
curl http://localhost:5000/api/disease/status?user_id=farmer123
```

### Analyze Progression

```bash
curl -X POST http://localhost:5000/api/disease/analyze-progression \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "farmer123"
  }'
```

---

## 🔧 Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size in `train_model.py`:
```python
BATCH_SIZE = 8  # or even 4
```

### Issue: Dataset Not Found

**Solution**: Update `DATASET_PATH` in `train_model.py`:
```python
DATASET_PATH = 'path/to/your/plant_disease'
```

### Issue: Slow Training

**Solutions**:
1. Use GPU if available
2. Reduce image size to (128, 128)
3. Use fewer LSTM units
4. Reduce sequence length to 3

### Issue: Low Accuracy

**Solutions**:
1. Train for more epochs
2. Unfreeze more CNN layers for fine-tuning
3. Increase data augmentation
4. Use class weights for imbalanced data

---

## 📈 Expected Results

After training, you should see:

- **Disease Classification Accuracy**: >85% (target: >90%)
- **Severity MAE**: <0.15 (target: <0.1)
- **Progression Rate MAE**: <0.05

If results are lower:
1. Train for more epochs
2. Fine-tune the CNN backbone
3. Adjust loss weights
4. Increase training data

---

## 🎨 Synthetic Data Generation

The model uses **synthetic temporal sequences** since real progression data isn't available:

1. **Progressive Degradation**: Simulates disease worsening over time
2. **Color Shifts**: Yellowing/browning effects
3. **Lesion Growth**: Expanding disease spots
4. **Texture Changes**: Wilting simulation

This approach allows training on standard datasets while maintaining temporal awareness.

---

## 🔮 Future Improvements

1. **Real Data Collection**: Build app to collect actual progression data from farmers
2. **Attention Mechanism**: Add attention layers to focus on important temporal features
3. **Multi-Modal Input**: Include weather, soil data
4. **Mobile Deployment**: Convert to TensorFlow Lite
5. **Explainability**: Enhanced Grad-CAM visualizations

---

## 📚 References

- [PlantVillage Dataset Paper](https://arxiv.org/abs/1511.08060)
- [CNN+LSTM for Sequence Learning](https://arxiv.org/abs/1411.4389)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Grad-CAM Visualization](https://arxiv.org/abs/1610.02391)

---

## 💡 Tips for Best Results

1. **Data Quality**: Ensure images are clear and well-lit
2. **Consistent Timing**: Upload images at similar times each day
3. **Same Plant**: Track the same plant/area over time
4. **Multiple Angles**: Capture different parts of the plant
5. **Good Lighting**: Natural daylight works best

---

## 🆘 Support

If you encounter issues:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review the [Implementation Plan](IMPLEMENTATION_PLAN.md)
3. Examine training logs in `logs/`
4. Check TensorBoard for training curves

---

## ✅ Checklist

Before training:
- [ ] Dataset downloaded and extracted
- [ ] Dependencies installed
- [ ] GPU configured (if available)
- [ ] Dataset path updated in `train_model.py`
- [ ] Sufficient disk space (~10GB for dataset + models)

After training:
- [ ] Model saved in `models/`
- [ ] Training history plotted
- [ ] Evaluation metrics reviewed
- [ ] API tested with sample data

---

**Ready to start? Run `python train_model.py` and let the training begin! 🚀**
