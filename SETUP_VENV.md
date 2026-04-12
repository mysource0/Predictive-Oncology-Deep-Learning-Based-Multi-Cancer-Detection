# 🐍 Setup Guide — Python venv (Virtual Environment)

Complete step-by-step guide to set up and run the **Predictive Oncology Multi-Cancer Detection System** using Python's built-in `venv` module — no Anaconda required.

---

## 📋 Prerequisites

Before starting, make sure you have:

1. **Python 3.9 – 3.11** installed
   - Download: https://www.python.org/downloads/
   - ⚠️ Python 3.12+ may have compatibility issues with TensorFlow 2.13
2. **pip** package manager (comes with Python)
3. **Git** (optional, for cloning the repo)
4. **Internet connection** (for installing packages and Gemini AI API)
5. **GPU** (optional but recommended for faster model training)

---

## Phase 1: Install Python

### If Python is NOT installed yet:

1. Download **Python 3.9.x** or **3.10.x** from https://www.python.org/downloads/
2. Run the installer
3. ⚠️ **IMPORTANT:** Check ✅ **"Add Python to PATH"** during installation
4. Complete the installation

### Verify Installation

Open **Command Prompt** (or PowerShell) and run:

```bash
python --version
```

Expected output: `Python 3.9.x` or `Python 3.10.x`

Also verify pip:

```bash
pip --version
```

Expected output: `pip 23.x.x from ...`

> ❌ If `python` is not recognized, Python was not added to PATH. Reinstall with the PATH checkbox enabled.

---

## Phase 2: Navigate to the Project Directory

Open **Command Prompt** or **PowerShell**:

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

## Phase 3: Create a Virtual Environment

```bash
python -m venv venv
```

This creates a `venv/` folder in your project directory containing an isolated Python environment.

> 💡 **Why use venv?** It keeps this project's packages separate from your system Python, avoiding version conflicts with other projects.

### Activate the Virtual Environment

**Windows (Command Prompt):**
```bash
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

> ⚠️ **PowerShell Execution Policy Error?** Run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### Verify Activation

After activation, you should see `(venv)` at the beginning of your prompt:
```
(venv) D:\finalYearProject\Predictive Oncology Deep Learning-Based Multi-Cancer Detection>
```

> 💡 **Important:** You must activate the virtual environment **every time** you open a new terminal to work on this project.

---

## Phase 4: Install Python Dependencies

With the virtual environment activated:

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

**Windows Command Prompt:**
```bash
copy .env.example .env
```

**PowerShell:**
```powershell
Copy-Item .env.example .env
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
5. Open `.env` in a text editor and replace `your-gemini-api-key-here` with your actual key

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
# Step 1: Open Command Prompt or PowerShell

# Step 2: Navigate to the project
cd "D:\finalYearProject\Predictive Oncology Deep Learning-Based Multi-Cancer Detection"

# Step 3: Activate the virtual environment
# Command Prompt:
venv\Scripts\activate
# PowerShell:
.\venv\Scripts\Activate.ps1

# Step 4: Run the app
python app.py

# Step 5: Open browser → http://localhost:5000
```

### Stopping the App

Press `Ctrl + C` in the terminal to stop the Flask server.

---

## 🔧 Common Issues & Fixes

### ❌ `python` command not found
- Python was not added to PATH during installation
- Reinstall Python and check ✅ **"Add Python to PATH"**
- Or use `py` instead of `python`: `py app.py`

### ❌ `venv\Scripts\activate` not recognized
- Make sure you're in the project root directory
- Make sure you created the venv: `python -m venv venv`

### ❌ PowerShell execution policy error
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### ❌ `ModuleNotFoundError`
```bash
# Make sure venv is activated (you should see (venv) in your prompt)
venv\Scripts\activate
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

### ❌ `pip install` fails with permission error
```bash
pip install -r requirements.txt --user
```
Or make sure your venv is activated (check for `(venv)` prefix in prompt).

### ❌ FutureWarning about Python version
These are harmless warnings from Google libraries. The app still works perfectly. Ignore them.

---

## 🗑️ Removing the Virtual Environment (if needed)

```bash
# Deactivate first
deactivate

# Delete the venv folder
# Command Prompt:
rmdir /s /q venv

# PowerShell:
Remove-Item -Recurse -Force venv
```

---

## 📝 Summary of Commands

| Action | Command |
|--------|---------|
| Create venv | `python -m venv venv` |
| Activate (CMD) | `venv\Scripts\activate` |
| Activate (PowerShell) | `.\venv\Scripts\Activate.ps1` |
| Install packages | `pip install -r requirements.txt` |
| Run the app | `python app.py` |
| Stop the app | `Ctrl + C` |
| Deactivate venv | `deactivate` |
| Delete venv | `rmdir /s /q venv` |
