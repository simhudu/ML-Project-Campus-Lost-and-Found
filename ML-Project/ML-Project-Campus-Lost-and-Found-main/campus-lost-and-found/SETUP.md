# Campus Lost & Found - Setup Guide

## Prerequisites

You need **Python 3.8+** installed on your system.

### Step 1: Install Python

1. Download Python from: https://www.python.org/downloads/
2. **IMPORTANT:** During installation, check the box: **"Add Python to PATH"**
3. Click "Install Now" and wait for completion
4. Restart your computer (or at least restart PowerShell)

### Step 2: Install Dependencies

Open PowerShell and run:

```powershell
cd "c:\Users\HP\OneDrive\Desktop\ML-Project-Campus-Lost-and-Found-main\ML-Project-Campus-Lost-and-Found-main\campus-lost-and-found"
python -m pip install -r requirements.txt
```

### Step 3: Run the Application

```powershell
python -m streamlit run app.py
```

The app will start on `http://localhost:8501` in your browser.

## Features

- **Login/Register:** Create an account and log in
- **Report Lost/Found Items:** Upload images with descriptions
- **AI Category Detection:** Auto-detect item categories (if model is trained)
- **Smart Matching:** Search for matching items using text and/or images
- **Hybrid Scoring:** 60% visual + 40% text-based matching

## Training the ML Model (Optional)

If you have a dataset, train the model:

```powershell
python train_model.py
```

The model will be saved to `modules/category_classifier.pkl`

## Troubleshooting

**Issue:** `streamlit: The term 'streamlit' is not recognized`
- **Solution:** Ensure Python is installed and added to PATH. Restart PowerShell.

**Issue:** `ModuleNotFoundError: No module named 'streamlit'`
- **Solution:** Run `python -m pip install -r requirements.txt`

**Issue:** AI detection returns None
- **Solution:** This is normal if the model hasn't been trained. Manual category selection is available.

---

Built with ❤️ using Streamlit + Python
