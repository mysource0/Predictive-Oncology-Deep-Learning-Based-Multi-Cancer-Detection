# 🧬 Predictive Oncology: Deep Learning-Based Multi-Cancer Detection

A Flask web application that combines **Convolutional Neural Networks (CNN)** with **Google Gemini AI** to detect multiple cancer types from medical images — including **Breast**, **Liver**, and **Skin** cancers.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-2.3-green?logo=flask)
![Gemini](https://img.shields.io/badge/Gemini_AI-2.5_Pro-purple?logo=google)

---

## 📋 Table of Contents

- [Features](#-features)
- [How It Works — Dual AI Pipeline](#-how-it-works--dual-ai-pipeline)
- [Supported Cancer Classes](#-supported-cancer-classes)
- [Project Structure](#-project-structure)
- [Setup — Local (Conda)](#-setup--local-conda-recommended)
- [Setup — Local (venv / pip)](#-setup--local-venv--pip)
- [Setup — Docker](#-setup--docker)
- [Usage Guide](#-usage-guide)
- [Dataset Structure](#-dataset-structure)
- [API Routes](#-api-routes)
- [Training Details](#-training-details)
- [Gemini AI Configuration](#-gemini-ai-configuration)
- [Troubleshooting](#-troubleshooting)
- [Security Notes](#-security-notes)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔐 **User Authentication** | Secure signup/signin with bcrypt password hashing |
| 🧠 **CNN Model Training** | Train a CNN directly from the web UI with real-time logs |
| 🔬 **Cancer Detection** | Upload an image and get a prediction with confidence score |
| 🤖 **Gemini AI Second Opinion** | Google Gemini 2.5 Pro validates the image and provides an independent diagnosis |
| 🚫 **Non-Medical Image Rejection** | Gemini automatically detects and rejects non-medical images (selfies, objects, etc.) |
| 📊 **6 Training Graphs** | Accuracy, Loss, Confusion Matrix, Per-Class Metrics, Overall Stats, Per-Class Accuracy |
| 🔄 **Retrain Support** | Retrain the model anytime with the click of a button |
| 📁 **Dynamic Classes** | Automatically discovers cancer classes from your dataset folders |
| 📋 **Model Info** | View model architecture, training status, and metadata |
| ⏳ **Loading Animations** | DNA helix loader with step-by-step progress indicators during analysis |
| 📱 **Responsive Design** | Works on desktop, tablet, and mobile screens |

---

## 🔬 How It Works — Dual AI Pipeline

When you upload a medical image, the system runs **two independent analyses**:

```
┌─────────────────────────────────────────────────────────┐
│                    User Uploads Image                    │
└─────────────────────┬───────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌─────────────────┐    ┌──────────────────────┐
│   CNN Model     │    │   Gemini 2.5 Pro AI  │
│   (TensorFlow)  │    │   (Google API)       │
├─────────────────┤    ├──────────────────────┤
│ • Classification│    │ • Image Validation   │
│ • Confidence %  │    │ • Medical Diagnosis  │
│ • Class Label   │    │ • Severity Rating    │
└────────┬────────┘    │ • Agreement Check    │
         │             │ • Recommendations    │
         │             └──────────┬───────────┘
         └───────────┬────────────┘
                     ▼
         ┌───────────────────────┐
         │   Combined Results    │
         │   • CNN Prediction    │
         │   • AI Second Opinion │
         │   • Severity Badge    │
         │   • Recommendations   │
         └───────────────────────┘
```

### Step 1 — Image Validation (Gemini)
Gemini checks if the uploaded image is a valid medical image. Non-medical images (selfies, food, landscapes, screenshots) are **automatically rejected** with a clear explanation.

### Step 2 — CNN Classification
The trained CNN model classifies the image into one of the supported cancer classes with a confidence percentage.

### Step 3 — AI Second Opinion (Gemini)
Gemini independently analyzes the image and provides:
- Its own diagnosis and detected condition
- Confidence level (0–100%)
- Severity rating (normal / low / moderate / high / critical)
- Whether it agrees or disagrees with the CNN prediction
- Detailed analysis notes with clinical observations
- Specific medical recommendations

---

## 🩺 Supported Cancer Classes

| Organ | Condition | Severity |
|-------|-----------|----------|
| 🩺 Breast | Normal | Healthy |
| ⚠️ Breast | Benign Tumor | Low — monitor regularly |
| 🔴 Breast | Malignant Tumor | High — consult oncologist |
| 🟢 Liver | Healthy | Healthy |
| 🟡 Liver | Hepatic Steatosis (Fatty Liver) | Moderate — lifestyle changes |
| 🧴 Skin | Healthy | Healthy |
| ⚠️ Skin | Dermatofibroma | Low — benign condition |
| 🔴 Skin | Seborrheic Keratosis | Low — non-cancerous growth |

---

## 📂 Project Structure

```
Predictive Oncology Deep Learning-Based Multi-Cancer Detection/
│
├── app.py                  # Main Flask app (routes, CNN training, Gemini integration)
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys, secrets)
├── .env.example            # Template for .env
├── .gitignore              # Git ignore rules
├── Dockerfile              # Docker container configuration
├── docker-compose.yml      # Docker Compose orchestration
├── README.md               # This file
│
├── model.h5                # Trained CNN model (generated after training)
├── model.json              # Model architecture JSON (generated)
├── users.db                # SQLite database (auto-created on first run)
│
├── static/                 # Static assets
│   ├── uploads/            # User-uploaded images (per-user timestamped folders)
│   ├── accuracy.png        # Training graph — accuracy (generated)
│   ├── loss.png            # Training graph — loss (generated)
│   ├── confusion_matrix.png
│   ├── metrics.png
│   ├── overall_stats.png
│   └── class_accuracy.png
│
├── templates/              # Jinja2 HTML templates
│   ├── home.html           # Dashboard with feature cards
│   ├── signup.html         # User registration
│   ├── signin.html         # User login
│   ├── detection.html      # Image upload, CNN + Gemini results
│   ├── train.html          # Model training page with 6 graphs
│   └── model_info.html     # Model architecture & status info
│
├── train/                  # Training dataset (one folder per class)
│   ├── Breast Normal/
│   ├── Breast Benign/
│   ├── Breast Malignant/
│   ├── Healthy Liver/
│   ├── Liver Hepatic_Steatosis/
│   ├── Skin Dermatofibroma/
│   ├── Skin Healthy/
│   └── Skin Seborrheic_Keratosis/
│
└── test/                   # Test dataset (same structure as train/)
```

---

## 🐍 Setup — Local (Conda) ⭐ Recommended

### Prerequisites
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- A [Google Gemini API key](https://aistudio.google.com/apikey) (free tier available)
- GPU support (optional but recommended for training)

### Step-by-Step

```bash
# 1. Open Anaconda Prompt

# 2. Create conda environment
conda create -n finalyearproject python=3.9 -y

# 3. Activate environment
conda activate finalyearproject

# 4. Navigate to project directory
cd "D:\finalYearProject\Predictive Oncology Deep Learning-Based Multi-Cancer Detection"

# 5. Install dependencies
pip install -r requirements.txt

# 6. Set up environment variables
copy .env.example .env
# Edit .env and add your Gemini API key:
#   GEMINI_API_KEY=your-api-key-here

# 7. Run the application
python app.py
```

### Access
Open your browser and go to: **http://localhost:5000**

### Quick Commands (after initial setup)
```bash
conda activate finalyearproject
cd "D:\finalYearProject\Predictive Oncology Deep Learning-Based Multi-Cancer Detection"
python app.py
```

---

## 🐍 Setup — Local (venv / pip)

### Prerequisites
- Python 3.9 – 3.11 installed
- `pip` package manager

### Step-by-Step

```bash
# 1. Navigate to project directory
cd "Predictive Oncology Deep Learning-Based Multi-Cancer Detection"

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set up environment variables
copy .env.example .env
# Edit .env and add your Gemini API key

# 6. Run the application
python app.py
```

### Access
Open your browser and go to: **http://localhost:5000**

---

## 🐳 Setup — Docker

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running

### Build & Run

```bash
# Navigate to project directory
cd "Predictive Oncology Deep Learning-Based Multi-Cancer Detection"

# Build and start (first time — may take 5-10 minutes)
docker-compose up --build

# Run in background (detached mode)
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Access
Open your browser and go to: **http://localhost:5000**

### Docker Useful Commands

```bash
# View running containers
docker ps

# Enter container shell (for debugging)
docker exec -it <container_name> bash

# Rebuild after code changes
docker-compose up --build

# Stop and remove everything (including volumes)
docker-compose down -v
```

---

## 📖 Usage Guide

### 1. Create an Account
- Visit `http://localhost:5000`
- Choose a username and password (min 6 characters)
- Click **📝 Register Account**

### 2. Sign In
- Go to the **Sign In** page
- Enter your credentials
- Click **🔐 Sign In**

### 3. Train the Model
- Click **🧠 Train Model** in the sidebar
- Click **Start Training**
- Wait for training to complete (~2–5 minutes depending on hardware)
- 6 training graphs will be generated and displayed
- To retrain later, click **🔄 Retrain Model**

### 4. Detect Cancer
- Click **🔬 Detection** in the sidebar
- Upload a medical image (PNG, JPG, JPEG, GIF, BMP — max 5MB)
- Click **🧬 Analyze Image**
- A loading animation shows the dual-analysis progress
- View combined results:
  - **CNN Model Analysis** — predicted class, confidence bar, recommendation
  - **Gemini AI Second Opinion** — independent diagnosis, severity, agreement badge, analysis notes, AI recommendation

### 5. View Model Info
- Click **📋 Model Info** to see:
  - Model training status and last trained timestamp
  - Discovered cancer classes
  - CNN architecture details
  - Supported conditions and dataset structure

---

## 📁 Dataset Structure

The model auto-discovers classes from the `train/` folder. Each subfolder name becomes a class label.

```
train/
├── Breast Normal/               # Healthy breast tissue images
├── Breast Benign/               # Non-cancerous breast tumor images
├── Breast Malignant/            # Cancerous breast tumor images
├── Healthy Liver/               # Normal liver scan images
├── Liver Hepatic_Steatosis/     # Fatty liver images
├── Skin Dermatofibroma/         # Benign skin condition images
├── Skin Healthy/                # Normal skin images
└── Skin Seborrheic_Keratosis/   # Non-cancerous skin growth images

test/
└── (same folder structure with test images)
```

> **Tip:** To add a new cancer type, simply create a new folder in both `train/` and `test/` with labeled images. The model will automatically detect and train on it.

---

## 🔌 API Routes

| Method | Route | Auth | Description |
|--------|-------|------|-------------|
| GET/POST | `/` | ❌ | Sign up page |
| GET/POST | `/signin` | ❌ | Sign in page |
| GET | `/home` | ✅ | Dashboard |
| GET/POST | `/detection` | ✅ | Upload image → CNN + Gemini analysis |
| GET | `/train` | ❌ | Train model page |
| GET | `/train?force=1` | ❌ | Force retrain the model |
| GET | `/model_info` | ✅ | View model status & architecture |
| GET | `/logout` | ✅ | Logout and clear session |

---

## 🧠 Training Details

### CNN Model Architecture
```
Input (150×150×3 RGB)
  → Conv2D(32, 3×3, ReLU) → MaxPooling2D
  → Conv2D(64, 3×3, ReLU) → MaxPooling2D
  → Flatten
  → Dense(128, ReLU)
  → Dropout(0.3)
  → Dense(num_classes, Softmax)
```

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Image Size | 150 × 150 px |
| Normalization | Pixel values / 255.0 |
| Epochs | 10 |
| Batch Size | 32 |
| Validation Split | 20% |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Dropout | 30% |

### Generated Graphs (after training)
1. 📈 **Accuracy** — Train vs Validation accuracy over epochs
2. 📉 **Loss** — Train vs Validation loss over epochs
3. 🎯 **Confusion Matrix** — True vs Predicted labels heatmap
4. 📊 **Per-Class Metrics** — Precision, Recall, F1-Score per class
5. 🏆 **Overall Performance** — Weighted Accuracy, Precision, Recall, F1
6. ✅ **Per-Class Accuracy** — Individual class accuracy horizontal bars

---

## 🤖 Gemini AI Configuration

The Gemini integration uses **Gemini 2.5 Pro** for medical image analysis.

### Getting an API Key
1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click **Create API Key**
4. Copy the key and paste it in your `.env` file:
   ```
   GEMINI_API_KEY=your-api-key-here
   ```

### What Gemini Does
| Step | Function |
|------|----------|
| **Validation** | Checks if the image is a valid medical image |
| **Rejection** | Rejects non-medical images with an explanation |
| **Diagnosis** | Provides an independent AI diagnosis |
| **Agreement** | Compares its diagnosis with the CNN prediction |
| **Severity** | Rates severity: normal / low / moderate / high / critical |
| **Notes** | Describes specific visual patterns observed |
| **Recommendation** | Provides targeted medical recommendation |

### Fallback Behavior
- If the Gemini API key is not configured, the system works with **CNN-only mode**
- If the Gemini API call fails, the CNN result is still shown with a warning badge
- Non-medical images are only rejected when Gemini is active

---

## 🔧 Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'flask'`
```bash
# Make sure your environment is activated
conda activate finalyearproject   # or: venv\Scripts\activate
pip install -r requirements.txt
```

### ❌ `SyntaxError` or app won't start
```bash
# Verify Python version (3.9+ required)
python --version

# Check for syntax errors
python -m py_compile app.py
```

### ❌ Model not found / "Train the model first"
- Go to `/train` and click **Start Training**
- Wait for training to complete before using Detection

### ❌ Training finds 0 classes
- Ensure `train/` folder exists with subfolders containing images
- Each subfolder name = class name

### ❌ File upload rejected
- Allowed formats: PNG, JPG, JPEG, GIF, BMP
- Max size: 5MB

### ❌ Gemini API errors
- Verify your API key in `.env` is correct
- Check that you have internet connectivity
- The free tier has rate limits — wait and retry if you hit them

### ❌ "Non-medical image" false positive
- Gemini may occasionally reject valid medical images of unusual formats
- Try a different image or temporarily disable Gemini by removing the API key

### ❌ Page hangs during training
- Training takes ~2–5 minutes depending on dataset size and hardware
- Do not refresh the page during training
- Check the terminal for progress logs

### ❌ GPU not detected (TensorFlow)
```bash
# Check if GPU is available
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install GPU support (if needed)
pip install tensorflow[and-cuda]
```

---

## 🔒 Security Notes

- Change `SECRET_KEY` in `.env` before deploying to production
- **Never commit `.env`** to version control (it's in `.gitignore`)
- The Gemini API key should be kept private
- Use HTTPS in production
- Consider a production WSGI server (Gunicorn/Waitress) for deployment
- Uploaded images are stored per-user with timestamps

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| Flask 2.3 | Web framework |
| TensorFlow 2.13 | CNN model training and inference |
| OpenCV | Image loading and preprocessing |
| NumPy | Array operations |
| Matplotlib | Training graph generation |
| Seaborn | Confusion matrix heatmap |
| scikit-learn | Metrics (precision, recall, F1) |
| google-generativeai | Gemini AI API client |
| Pillow | Image handling for Gemini |
| python-dotenv | Environment variable loading |
| Werkzeug | Password hashing & file security |

---

## 📄 License

This project is for **educational and research purposes** only.

> ⚠️ **Disclaimer:** The predictions made by this system are NOT a substitute for professional medical diagnosis. Always consult qualified healthcare professionals and licensed oncologists for actual medical diagnosis and treatment decisions.
