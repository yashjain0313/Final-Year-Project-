# Quick Test Training Configuration

## 🚀 Quick Test (Recommended First)

To test if everything works before the full training, modify `train_model.py`:

```python
# Line 419-421: Change these values
BATCH_SIZE = 4      # Smaller batch for CPU
EPOCHS = 5          # Just 5 epochs for testing
LEARNING_RATE = 0.0001
```

This will:
- ✅ Train for only ~2-3 hours on CPU
- ✅ Verify the pipeline works
- ✅ Generate sample outputs
- ✅ Then you can run full training (50 epochs) later

---

## 📊 Full Training (After Test)

Once the quick test works, change back to:

```python
BATCH_SIZE = 8      # Or 16 if you have more RAM
EPOCHS = 50         # Full training
LEARNING_RATE = 0.0001
```

---

## 💡 Google Colab Alternative (FREE GPU!)

If you want faster training:

1. **Upload your code to Google Drive**
2. **Open Google Colab**: https://colab.research.google.com
3. **Enable GPU**: Runtime → Change runtime type → GPU
4. **Mount Drive and run training**

This will train in **2-4 hours** instead of 12-24!

---

## ⚡ Commands

### Start Quick Test (5 epochs):
```powershell
# Edit train_model.py first (change EPOCHS to 5, BATCH_SIZE to 4)
python train_model.py
```

### Monitor Training:
```powershell
# In a new terminal
tensorboard --logdir=logs/
# Open http://localhost:6006
```

### Check Progress:
Training will show:
- Current epoch
- Loss values
- Accuracy metrics
- ETA for completion

---

## 📝 What to Expect

### During Training:
```
Epoch 1/5
████████████████ 1234/1234 [======] - 45min - loss: 2.5 - accuracy: 0.45
Epoch 2/5
████████████████ 1234/1234 [======] - 43min - loss: 1.8 - accuracy: 0.62
...
```

### After Training:
- ✅ `models/disease_progression_best.h5` - Best model
- ✅ `models/disease_progression_final.h5` - Final model
- ✅ `training_history.png` - Training curves
- ✅ `confusion_matrix.png` - Classification results
- ✅ `regression_plots.png` - Severity/progression plots

---

**Ready to start? Just run `python train_model.py`!** 🚀
