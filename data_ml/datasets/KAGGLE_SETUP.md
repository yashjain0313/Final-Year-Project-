# Kaggle API Setup Guide

## 🔑 Setting Up Kaggle Credentials

You need to set up Kaggle API credentials before downloading datasets.

### Step 1: Get Your Kaggle API Token

1. **Go to Kaggle**: https://www.kaggle.com
2. **Sign in** (or create an account if you don't have one)
3. **Click on your profile picture** (top right)
4. **Select "Settings"**
5. **Scroll down to "API" section**
6. **Click "Create New Token"**
7. This will download a file called **`kaggle.json`**

### Step 2: Place the kaggle.json File

**For Windows:**

Place the `kaggle.json` file in:
```
C:\Users\jeeva\.kaggle\kaggle.json
```

**Steps:**
1. Create the `.kaggle` folder if it doesn't exist:
   ```powershell
   mkdir $env:USERPROFILE\.kaggle
   ```

2. Move the downloaded `kaggle.json` to this folder:
   ```powershell
   # If kaggle.json is in Downloads:
   Move-Item $env:USERPROFILE\Downloads\kaggle.json $env:USERPROFILE\.kaggle\kaggle.json
   ```

3. Verify it's there:
   ```powershell
   Get-Content $env:USERPROFILE\.kaggle\kaggle.json
   ```

### Step 3: Download the Dataset

Once credentials are set up, run:

```powershell
cd "C:\Users\jeeva\Final Year Projects\Final-Year-Project-\data_ml\datasets"
kaggle datasets download -d emmarex/plantdisease
```

### Step 4: Extract the Dataset

```powershell
# Extract the zip file
Expand-Archive -Path plantdisease.zip -DestinationPath plant_disease -Force
```

---

## 🔄 Alternative: Manual Download

If you prefer to download manually:

1. **Visit**: https://www.kaggle.com/datasets/emmarex/plantdisease
2. **Click the "Download" button** (you may need to sign in)
3. **Save to**: `C:\Users\jeeva\Final Year Projects\Final-Year-Project-\data_ml\datasets\`
4. **Extract** the zip file to a folder named `plant_disease`

---

## ✅ Verify Dataset

After extraction, verify the dataset structure:

```powershell
# Check if dataset exists
Get-ChildItem "C:\Users\jeeva\Final Year Projects\Final-Year-Project-\data_ml\datasets\plant_disease"
```

You should see folders like:
- `Pepper__bell___Bacterial_spot`
- `Pepper__bell___healthy`
- `Potato___Early_blight`
- `Potato___Late_blight`
- `Tomato___Bacterial_spot`
- etc. (38 classes total)

---

## 🚀 Next Steps

Once the dataset is downloaded and extracted:

1. **Update the dataset path** in `train_model.py`:
   ```python
   DATASET_PATH = '../../datasets/plant_disease'
   ```

2. **Run the setup test**:
   ```powershell
   cd "..\notebooks\disease_progression"
   python test_setup.py
   ```

3. **Start training**:
   ```powershell
   python train_model.py
   ```

---

## 🐛 Troubleshooting

### "kaggle: command not found" (after installation)

**Solution**: Close and reopen your terminal/PowerShell to refresh the PATH.

### "401 Unauthorized"

**Solution**: Your `kaggle.json` credentials are incorrect or not in the right location.
- Verify the file is at: `C:\Users\jeeva\.kaggle\kaggle.json`
- Re-download the token from Kaggle settings

### "403 Forbidden"

**Solution**: You need to accept the dataset's terms and conditions.
- Visit: https://www.kaggle.com/datasets/emmarex/plantdisease
- Click "Download" to accept terms
- Then try the kaggle command again

---

## 📝 Quick Commands Summary

```powershell
# 1. Create .kaggle folder
mkdir $env:USERPROFILE\.kaggle

# 2. Move kaggle.json (if in Downloads)
Move-Item $env:USERPROFILE\Downloads\kaggle.json $env:USERPROFILE\.kaggle\kaggle.json

# 3. Download dataset
cd "C:\Users\jeeva\Final Year Projects\Final-Year-Project-\data_ml\datasets"
kaggle datasets download -d emmarex/plantdisease

# 4. Extract
Expand-Archive -Path plantdisease.zip -DestinationPath plant_disease -Force

# 5. Verify
Get-ChildItem plant_disease

# 6. Test setup
cd "..\notebooks\disease_progression"
python test_setup.py
```

---

**You're almost there! Just set up the Kaggle credentials and you'll be ready to train!** 🚀
