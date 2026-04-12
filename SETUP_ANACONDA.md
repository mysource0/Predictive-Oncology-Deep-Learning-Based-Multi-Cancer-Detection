# 🐍 Setup Guide — Anaconda / Miniconda

Complete step-by-step guide to set up and run the **Predictive Oncology Multi-Cancer Detection System** using Anaconda or Miniconda.

---

## 📋 Prerequisites

Before starting, make sure you have:

1. **Anaconda** or **Miniconda** installed
   - Download Anaconda: https://www.anaconda.com/download
   - Download Miniconda (lightweight): https://docs.conda.io/en/latest/miniconda.html
2. **Git** (optional, for cloning the repo)
3. **Internet connection** (for installing packages and Gemini AI API)
4. **GPU** (optional but recommended for faster model training)

---

## Phase 1: Install Anaconda

### If Anaconda is NOT installed yet:

1. Download the **Anaconda Installer** from https://www.anaconda.com/download
2. Run the installer
3. During installation:
   - ✅ Check **"Add Anaconda to my PATH environment variable"** (recommended)
   - ✅ Check **"Register Anaconda as my default Python"**
4. Complete the installation
5. Restart your terminal / command prompt

### Verify Installation

Open **Anaconda Prompt** (search in Start Menu) and run:

```bash
conda --version
```

Expected output: `conda 24.x.x` (or similar)

---

## Phase 2: Create the Conda Environment

Open **Anaconda Prompt** and run the following commands:

```bash
# Create a new environment named 'finalyearproject' with Python 3.9
conda create -n finalyearproject python=3.9 -y
```

> ⚠️ **Why Python 3.9?** TensorFlow 2.13 has the best compatibility with Python 3.9. Newer versions may have issues with some dependencies.

### Activate the Environment

```bash
conda activate finalyearproject
```

You should see `(finalyearproject)` at the beginning of your prompt:
```
(finalyearproject) C:\Users\YourName>
```

> 💡 **Important:** You must activate this environment **every time** you open a new terminal to work on this project.

---

## Phase 3: Navigate to the Project Directory

```bash
cd "D:\finalYearProject\Predictive Oncology Deep Learning-Based Multi-Cancer Detection"
```

If your project is in a different location, adjust the path accordingly.

### Verify you're in the right directory:

```bash
dir app.py
```

You should see `app.py` listed. If not, you're in the wrong directory.

---

## Phase 4: Install Python Dependencies

With the conda environment activated and in the project directory:

```bash
pip install -r requirements.txt
```

This installs all required packages:
- Flask (web framework)
- TensorFlow (deep learning)
- OpenCV (image processing)
- NumPy, Matplotlib, Seaborn (data & visualization)
- scikit-learn (metrics)
- google-generativeai (Gemini AI)
- Pillow (image handling)
- python-dotenv (environment variables)

### Verify Installation

```bash
python -c "import flask; import tensorflow; import cv2; print('All packages installed successfully!')"
```

> ⏳ This step may take **5–10 minutes** depending on your internet speed, especially for TensorFlow.

---

## Phase 5: Configure Environment Variables

### Option A: Copy the template (recommended)

```bash
copy .env.example .env
```

### Option B: Create manually

Create a file named `.env` in the project root with:

```env
# Flask Configuration
SECRET_KEY=your-secret-key-change-this-in-production
FLASK_ENV=development
FLASK_DEBUG=False

# Database
DATABASE_URL=sqlite:///users.db

# Upload Settings
MAX_UPLOAD_SIZE_MB=5
ALLOWED_EXTENSIONS=png,jpg,jpeg,gif,bmp

# Gemini AI API Key (get from https://aistudio.google.com/apikey)
GEMINI_API_KEY=your-gemini-api-key-here
```

### Get a Gemini API Key (Required for AI Second Opinion)

1. Visit https://aistudio.google.com/apikey
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key
5. Open `.env` and replace `your-gemini-api-key-here` with your actual key

> 💡 The app works **without** a Gemini API key, but the AI second opinion and non-medical image rejection features will be disabled.

---

## Phase 6: Verify the Dataset

Ensure your training and test data folders exist:

```bash
dir train
dir test
```

You should see folders like:
```
Breast Benign/
Breast Malignant/
Breast Normal/
Healthy Liver/
Liver Hepatic_Steatosis/
Skin Dermatofibroma/
Skin Healthy/
Skin Seborrheic_Keratosis/
```

Each folder should contain medical images (PNG, JPG, etc.)

---

## Phase 7: Run the Application

```bash
python app.py
```

### Expected Output

```
 * Serving Flask app 'app'
 * Debug mode: off
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

### Open in Browser

Navigate to: **http://localhost:5000**

---

## Phase 8: First-Time Usage

### 1. Register an Account
- You'll land on the **Sign Up** page
- Enter a username and password (min 6 characters)
- Click **Register Account**

### 2. Sign In
- Enter your credentials
- Click **Sign In**

### 3. Train the Model
- Click **🧠 Train Model** in the sidebar
- Click **Start Training**
- Wait ~2–5 minutes for training to complete
- 6 graphs will be generated showing model performance

### 4. Detect Cancer
- Click **🔬 Detection** in the sidebar
- Upload a medical image
- Click **🧬 Analyze Image**
- View CNN prediction + Gemini AI second opinion

---

## 🔄 Running the App (Every Time After Setup)

Every time you want to start the application:

```bash
# Step 1: Open Anaconda Prompt

# Step 2: Activate the environment
conda activate finalyearproject

# Step 3: Navigate to the project
cd "D:\finalYearProject\Predictive Oncology Deep Learning-Based Multi-Cancer Detection"

# Step 4: Run the app
python app.py

# Step 5: Open browser → http://localhost:5000
```

### Stopping the App

Press `Ctrl + C` in the terminal to stop the Flask server.

---

## 🔧 Common Issues & Fixes

### ❌ `conda` command not found
- Close and reopen Anaconda Prompt
- Or add Anaconda to your PATH manually

### ❌ `ModuleNotFoundError`
```bash
# Make sure the environment is activated
conda activate finalyearproject
pip install -r requirements.txt
```

### ❌ TensorFlow GPU warnings
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
If no GPU is found, TensorFlow will use CPU (slower training but still works).

### ❌ Port 5000 already in use
Another application is using port 5000. Either:
- Close the other application
- Or change the port in `app.py`: `app.run(port=5001)`

### ❌ Model not found error
- You need to train the model first via the **Train Model** page

### ❌ FutureWarning about Python version
These are harmless warnings from Google libraries. The app still works perfectly. Ignore them.

---

## 🗑️ Removing the Environment (if needed)

```bash
# Deactivate first
conda deactivate

# Remove the environment completely
conda remove -n finalyearproject --all -y
```

---

## 📝 Summary of Commands

| Action | Command |
|--------|---------|
| Create environment | `conda create -n finalyearproject python=3.9 -y` |
| Activate environment | `conda activate finalyearproject` |
| Install packages | `pip install -r requirements.txt` |
| Run the app | `python app.py` |
| Stop the app | `Ctrl + C` |
| Deactivate environment | `conda deactivate` |
| Remove environment | `conda remove -n finalyearproject --all -y` |
