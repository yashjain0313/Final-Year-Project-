# 📦 Manual Dataset Extraction Guide

## Current Status
✅ You've downloaded the dataset from Kaggle via Chrome
⚠️ Dataset needs to be extracted to the correct location

---

## 🎯 Step-by-Step Extraction

### Step 1: Find Your Downloaded File

The zip file should be in one of these locations:
- `C:\Users\jeeva\Downloads\plantdisease.zip`
- `C:\Users\jeeva\Downloads\archive.zip`
- `C:\Users\jeeva\Downloads\plant-disease-dataset.zip`

### Step 2: Extract Using PowerShell

**Option A: If file is named `plantdisease.zip`**
```powershell
cd "C:\Users\jeeva\Final Year Projects\Final-Year-Project-\data_ml\datasets"
Expand-Archive -Path "$env:USERPROFILE\Downloads\plantdisease.zip" -DestinationPath "plant_disease" -Force
```

**Option B: If file is named `archive.zip`**
```powershell
cd "C:\Users\jeeva\Final Year Projects\Final-Year-Project-\data_ml\datasets"
Expand-Archive -Path "$env:USERPROFILE\Downloads\archive.zip" -DestinationPath "plant_disease" -Force
```

**Option C: Manual Extraction (Windows Explorer)**
1. Open File Explorer
2. Navigate to `C:\Users\jeeva\Downloads\`
3. Find the downloaded zip file (plantdisease.zip or archive.zip)
4. **Right-click** on the zip file
5. Select **"Extract All..."**
6. Set destination to: `C:\Users\jeeva\Final Year Projects\Final-Year-Project-\data_ml\datasets\plant_disease`
7. Click **"Extract"**

### Step 3: Verify Extraction

Run this command to check if extraction worked:
```powershell
Get-ChildItem "C:\Users\jeeva\Final Year Projects\Final-Year-Project-\data_ml\datasets\plant_disease" | Select-Object -First 10
```

You should see folders like:
```
Pepper__bell___Bacterial_spot
Pepper__bell___healthy
Potato___Early_blight
Potato___Late_blight
Potato___healthy
Tomato___Bacterial_spot
Tomato___Early_blight
Tomato___Late_blight
...and more (38 classes total)
```

### Step 4: Check Image Count

```powershell
(Get-ChildItem "C:\Users\jeeva\Final Year Projects\Final-Year-Project-\data_ml\datasets\plant_disease" -Recurse -File -Include *.jpg,*.jpeg,*.png).Count
```

This should show approximately **54,000 images**.

---

## 🔍 Troubleshooting

### Issue: "Extraction creates nested folders"

Sometimes extraction creates:
```
plant_disease/
  └── PlantVillage/
      └── [actual class folders]
```

**Fix**: Move the contents up one level:
```powershell
cd "C:\Users\jeeva\Final Year Projects\Final-Year-Project-\data_ml\datasets"
Move-Item "plant_disease\PlantVillage\*" "plant_disease\" -Force
Remove-Item "plant_disease\PlantVillage" -Recurse
```

### Issue: "Can't find the zip file"

Check your Downloads folder:
```powershell
Get-ChildItem "$env:USERPROFILE\Downloads\*.zip" | Where-Object {$_.Name -like "*plant*" -or $_.Name -like "*archive*"}
```

---

## ✅ After Successful Extraction

### Run the setup test again:
```powershell
cd "C:\Users\jeeva\Final Year Projects\Final-Year-Project-\data_ml\notebooks\disease_progression"
python test_setup.py
```

You should now see:
```
✅ Dataset found!
   Path: ../../datasets/plant_disease
   Classes: 38
   Total images: ~54000
```

### If all checks pass, start training:
```powershell
python train_model.py
```

---

## 🚀 Quick Commands (Copy-Paste Ready)

```powershell
# Navigate to datasets folder
cd "C:\Users\jeeva\Final Year Projects\Final-Year-Project-\data_ml\datasets"

# Extract (try this first - adjust filename if needed)
Expand-Archive -Path "$env:USERPROFILE\Downloads\archive.zip" -DestinationPath "plant_disease" -Force

# Verify extraction
Get-ChildItem "plant_disease" | Select-Object -First 10

# Count images
(Get-ChildItem "plant_disease" -Recurse -File -Include *.jpg,*.jpeg,*.png).Count

# Go back to disease_progression folder
cd "..\notebooks\disease_progression"

# Run test again
python test_setup.py

# If all passes, train!
python train_model.py
```

---

**Need help? Let me know which step you're stuck on!** 🚀
