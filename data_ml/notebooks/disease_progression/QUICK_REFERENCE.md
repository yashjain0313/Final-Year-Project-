# Disease Progression Detection - Quick Reference

## 🚀 Quick Start Commands

```bash
# 1. Install dependencies
pip install tensorflow keras opencv-python numpy pandas matplotlib seaborn scikit-learn pillow

# 2. Download dataset
cd data_ml/datasets
kaggle datasets download -d emmarex/plantdisease
unzip plantdisease.zip -d plant_disease

# 3. Verify setup
cd ../notebooks/disease_progression
python test_setup.py

# 4. Train model
python train_model.py

# 5. Monitor training
tensorboard --logdir=logs/

# 6. Run API
cd ../../../backend
python disease_progression_api.py
```

---

## 📁 File Structure

```
disease_progression/
├── IMPLEMENTATION_PLAN.md    # Detailed architecture & strategy
├── README.md                  # Setup guide
├── SUMMARY.md                 # Complete explanation
├── QUICK_REFERENCE.md         # This file
├── disease_progression_model.py  # Model architecture
├── train_model.py             # Training script
├── test_setup.py              # Setup verification
├── models/                    # Saved models (created during training)
├── logs/                      # TensorBoard logs
└── ../../datasets/plant_disease/  # Dataset location
```

---

## 🎯 Key Concepts

### CNN+LSTM Hybrid
- **CNN**: Extracts visual features from each image
- **LSTM**: Processes temporal sequence (Day 1 → Day 2 → Day 3)
- **Result**: Understands both what disease looks like AND how it progresses

### Multi-Task Learning
Three outputs from one model:
1. **Disease Classification**: Which disease? (38 classes)
2. **Severity Score**: How bad? (0-1 scale)
3. **Progression Rate**: How fast worsening? (per day)

### Synthetic Temporal Data
Since no real temporal datasets exist:
- Take single disease images
- Generate progression sequences using augmentation
- Simulate: color shifts, lesion growth, wilting

---

## 📊 Model Configuration

```python
# Default settings (in train_model.py)
IMAGE_SIZE = (224, 224)
SEQUENCE_LENGTH = 5          # Number of days
BATCH_SIZE = 16              # Reduce if GPU memory issues
EPOCHS = 50
LEARNING_RATE = 0.0001
CNN_BACKBONE = 'efficientnet'  # or 'resnet50'
LSTM_UNITS = [256, 128]
```

---

## 🔧 Common Modifications

### Reduce Memory Usage
```python
BATCH_SIZE = 8  # or even 4
IMAGE_SIZE = (128, 128)
LSTM_UNITS = [128, 64]
```

### Speed Up Training
```python
CNN_BACKBONE = 'resnet50'  # Faster than efficientnet
EPOCHS = 30
```

### Improve Accuracy
```python
EPOCHS = 100
# Unfreeze more CNN layers
model_builder.unfreeze_backbone(num_layers=40)
```

---

## 📈 Expected Training Time

| Hardware | Batch Size | Time (50 epochs) |
|----------|------------|------------------|
| RTX 3080 | 32 | 2-3 hours |
| RTX 2070 | 16 | 3-4 hours |
| GTX 1060 | 8 | 6-8 hours |
| CPU only | 4 | 18-24 hours |

---

## 🎯 Target Metrics

| Metric | Target | Acceptable |
|--------|--------|------------|
| Disease Accuracy | >90% | >85% |
| Severity MAE | <0.10 | <0.15 |
| Progression MAE | <0.05 | <0.08 |

---

## 🌐 API Endpoints

### 1. Upload Day Image
```bash
POST /api/disease/upload-day
{
  "user_id": "farmer123",
  "day": 1,
  "image": "data:image/jpeg;base64,..."
}
```

### 2. Check Status
```bash
GET /api/disease/status?user_id=farmer123
```

### 3. Analyze Progression
```bash
POST /api/disease/analyze-progression
{
  "user_id": "farmer123"
}
```

---

## 🐛 Troubleshooting

### Out of Memory
```python
# Reduce batch size
BATCH_SIZE = 4

# Or reduce image size
IMAGE_SIZE = (128, 128)
```

### Dataset Not Found
```python
# Update path in train_model.py
DATASET_PATH = 'path/to/your/plant_disease'
```

### Slow Training
```python
# Use ResNet instead of EfficientNet
CNN_BACKBONE = 'resnet50'

# Reduce sequence length
SEQUENCE_LENGTH = 3
```

### Low Accuracy
```python
# Train longer
EPOCHS = 100

# Fine-tune CNN
model_builder.unfreeze_backbone(num_layers=40)
model_builder.compile_model(learning_rate=0.00005)
```

---

## 📚 Important Files

### Training Outputs
- `models/disease_progression_best.h5` - Best model
- `models/disease_progression_final.h5` - Final model
- `models/class_names.npy` - Class labels
- `training_history.png` - Training curves
- `confusion_matrix.png` - Classification results
- `regression_plots.png` - Severity/progression accuracy

### Logs
- `logs/disease_progression/` - TensorBoard logs
- `logs/disease_progression_training.csv` - Training metrics

---

## 💡 Tips

### Before Training
1. ✅ Run `test_setup.py` first
2. ✅ Check GPU is detected
3. ✅ Verify dataset is downloaded
4. ✅ Ensure 10GB+ free disk space

### During Training
1. 📊 Monitor TensorBoard
2. 👀 Watch for overfitting (val_loss increasing)
3. ⏰ Use early stopping (patience=10)
4. 💾 Best model auto-saved

### After Training
1. 📈 Check confusion matrix
2. 🎯 Verify severity/progression plots
3. 🧪 Test with sample images
4. 🚀 Deploy to API

---

## 🔍 Model Interpretability

### Grad-CAM Visualization
```python
from disease_progression_model import GradCAMVisualizer

visualizer = GradCAMVisualizer(model)
heatmap = visualizer.generate_heatmap(image, class_idx)
overlay = visualizer.overlay_heatmap(image, heatmap)

plt.imshow(overlay)
plt.show()
```

Shows which image regions influenced the prediction.

---

## 📞 Support Checklist

If something goes wrong:

1. ✅ Check error message carefully
2. ✅ Review [Troubleshooting](#-troubleshooting) section
3. ✅ Verify dataset path is correct
4. ✅ Check GPU memory usage
5. ✅ Look at TensorBoard logs
6. ✅ Try with smaller batch size
7. ✅ Check Python/TensorFlow versions

---

## 🎓 Key Papers & Resources

1. **PlantVillage Dataset**: https://arxiv.org/abs/1511.08060
2. **CNN+LSTM**: https://arxiv.org/abs/1411.4389
3. **EfficientNet**: https://arxiv.org/abs/1905.11946
4. **Grad-CAM**: https://arxiv.org/abs/1610.02391

---

## ✅ Pre-Training Checklist

- [ ] TensorFlow installed
- [ ] Dataset downloaded
- [ ] GPU detected (optional but recommended)
- [ ] `test_setup.py` passes all checks
- [ ] Dataset path updated in `train_model.py`
- [ ] 10GB+ disk space available
- [ ] TensorBoard ready to monitor

---

## 🚀 Deployment Checklist

- [ ] Model trained successfully
- [ ] Accuracy >85%
- [ ] Model saved to `models/`
- [ ] API tested with sample data
- [ ] Frontend UI ready for multi-day upload
- [ ] Grad-CAM visualization working
- [ ] Error handling implemented

---

## 📊 Training Progress Indicators

### Good Signs ✅
- Validation loss decreasing
- Accuracy increasing steadily
- No huge gap between train/val metrics
- Severity MAE decreasing

### Warning Signs ⚠️
- Validation loss increasing (overfitting)
- Training loss stuck (learning rate too low)
- NaN losses (learning rate too high)
- Very slow progress (batch size too small)

---

## 🎯 Success Metrics

### Minimum Viable Product
- ✅ Model trains without errors
- ✅ Disease accuracy >80%
- ✅ API accepts uploads
- ✅ Returns predictions

### Production Ready
- ✅ Disease accuracy >90%
- ✅ Severity MAE <0.1
- ✅ Inference <3 seconds
- ✅ Grad-CAM working
- ✅ User-friendly frontend

---

**Quick Start**: `python test_setup.py && python train_model.py`

**Monitor**: `tensorboard --logdir=logs/`

**Deploy**: Update `backend/app.py` with disease progression routes

---

Good luck! 🌱🚀
