# 🧬 Predictive Oncology: Deep Learning-Based Multi-Cancer Detection

A Flask web application that uses **Convolutional Neural Networks (CNN)** to detect multiple cancer types from medical images — including **Breast**, **Liver**, and **Skin** cancers.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-2.3-green?logo=flask)

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Setup — Local (Conda)](#-setup--local-conda-recommended)
- [Setup — Local (venv / pip)](#-setup--local-venv--pip)
- [Setup — Docker](#-setup--docker)
- [Usage Guide](#-usage-guide)
- [Dataset Structure](#-dataset-structure)
- [API Routes](#-api-routes)
- [Training Details](#-training-details)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔐 **User Authentication** | Secure signup/signin with bcrypt password hashing |
| 🧠 **Model Training** | Train a CNN directly from the web UI with real-time logs |
| 🔬 **Cancer Detection** | Upload an image and get a prediction with confidence score |
| 📊 **6 Training Graphs** | Accuracy, Loss, Confusion Matrix, Per-Class Metrics, Overall Stats, Per-Class Accuracy |
| 🔄 **Retrain Support** | Retrain the model anytime with the click of a button |
| 📁 **Dynamic Classes** | Automatically discovers cancer classes from your dataset folders |
| 📋 **Model Info** | View model training status and metadata |

---

## 📂 Project Structure

```
Predictive Oncology Deep Learning-Based Multi-Cancer Detection/
│
├── app.py                  # Main Flask application (routes, training, detection)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore rules
├── model.h5                # Trained model file (generated after training)
├── users.db                # SQLite database (auto-created on first run)
│
├── static/                 # Static assets
│   ├── uploads/            # User-uploaded images (per-user folders)
│   ├── accuracy.png        # Training accuracy graph (generated)
│   ├── loss.png            # Training loss graph (generated)
│   ├── confusion_matrix.png
│   ├── metrics.png
│   ├── overall_stats.png
│   └── class_accuracy.png
│
├── templates/              # HTML templates (Jinja2)
│   ├── home.html           # Dashboard / landing page
│   ├── signup.html         # User registration
│   ├── signin.html         # User login
│   ├── detection.html      # Image upload & prediction
│   ├── train.html          # Model training page with graphs
│   └── model_info.html     # Model status information
│
├── train/                  # Training dataset (organized by class folders)
│   ├── Breast  Normal/
│   ├── Breast Benign/
│   ├── Breast Malignant/
│   ├── Healthy Liver/
│   ├── Liver Hepatic_Steatosis/
│   ├── Skin Dermatofibroma/
│   ├── Skin Healthy/
│   └── Skin seborrheic_keratosis/
│
└── test/                   # Test dataset (same structure as train/)
```

---

## 🐍 Setup — Local (Conda) ⭐ Recommended

### Prerequisites
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- GPU support (optional but recommended for training)

### Step-by-Step

```bash
# 1. Open Anaconda Prompt

# 2. Create conda environment
conda create -n finalyearproject python=3.10 -y

# 3. Activate environment
conda activate finalyearproject

# 4. Navigate to project directory
cd "D:\finalYearProject\Predictive Oncology Deep Learning-Based Multi-Cancer Detection"

# 5. Install dependencies
pip install -r requirements.txt

# 6. (Optional) Create .env file for custom secret key
copy .env.example .env
# Edit .env and set a secure SECRET_KEY

# 7. Run the application
python app.py
```

### Access
Open your browser and go to: **http://localhost:5000**

### Quick Commands (after initial setup)
```bash
# Every time you want to run:
conda activate finalyearproject
cd "D:\finalYearProject\Predictive Oncology Deep Learning-Based Multi-Cancer Detection"
python app.py
```

---

## 🐍 Setup — Local (venv / pip)

### Prerequisites
- Python 3.8 – 3.11 installed
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

# 5. (Optional) Create .env file
copy .env.example .env

# 6. Run the application
python app.py
```

### Access
Open your browser and go to: **http://localhost:5000**

---

## 🐳 Setup — Docker

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running

### Step 1: Create the Dockerfile

Create a file called `Dockerfile` in the project root:

```dockerfile
FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create upload directory
RUN mkdir -p static/uploads

EXPOSE 5000

ENV FLASK_ENV=production
ENV SECRET_KEY=change-this-to-a-secure-key

CMD ["python", "app.py"]
```

### Step 2: Create docker-compose.yml

Create a file called `docker-compose.yml` in the project root:

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./train:/app/train:ro          # Training data (read-only)
      - ./test:/app/test:ro            # Test data (read-only)
      - uploads_data:/app/static/uploads  # Persistent uploads
      - model_data:/app/model_data       # Persistent model
    environment:
      - SECRET_KEY=your-secure-secret-key
      - FLASK_ENV=production
    restart: unless-stopped

volumes:
  uploads_data:
  model_data:
```

### Step 3: Build & Run

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

# View logs in real-time
docker-compose logs -f

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
- Fill in a username and password (min 6 characters)
- Click **Sign Up**

### 2. Sign In
- Go to the **Sign In** page
- Enter your credentials

### 3. Train the Model
- Click **🧠 Train Model** in the sidebar
- Click **Start Training** button
- Wait for training to complete (~2–5 minutes depending on hardware)
- 6 training graphs will be generated and displayed
- To retrain later, click **🔄 Retrain Model**

### 4. Detect Cancer
- Click **🔬 Detection** in the sidebar
- Upload a medical image (PNG, JPG, JPEG, GIF, BMP — max 5MB)
- The model will classify the image and show:
  - **Predicted class** (e.g., "Breast Malignant")
  - **Confidence score** (percentage)
  - **Medical suggestion**

### 5. View Model Info
- Click **📋 Model Info** to see training status and metadata

---

## 📁 Dataset Structure

The model auto-discovers classes from the `train/` folder. Each subfolder = one class.

```
train/
├── Breast  Normal/          # ~187 images
├── Breast Benign/           # ~204 images
├── Breast Malignant/        # ~200 images
├── Healthy Liver/           # ~200 images
├── Liver Hepatic_Steatosis/ # ~191 images
├── Skin Dermatofibroma/     # ~95 images
├── Skin Healthy/            # ~143 images
└── Skin seborrheic_keratosis/ # ~18 images

test/
└── (same folder structure with test images)
```

> **Tip:** To add a new cancer type, simply create a new folder in both `train/` and `test/` with images. The model will automatically detect and train on it.

---

## 🔌 API Routes

| Method | Route | Auth | Description |
|--------|-------|------|-------------|
| GET/POST | `/` | ❌ | Sign up page |
| GET/POST | `/signin` | ❌ | Sign in page |
| GET | `/home` | ✅ | Dashboard |
| GET/POST | `/detection` | ✅ | Upload image & get prediction |
| GET | `/train` | ❌ | Train model page |
| GET | `/train?force=1` | ❌ | Force retrain the model |
| GET | `/model_info` | ✅ | View model status |
| GET | `/logout` | ✅ | Logout |

---

## 🧠 Training Details

### Model Architecture (CNN)
```
Input (150×150×3)
  → Conv2D(32, 3×3, ReLU) → MaxPooling2D
  → Conv2D(64, 3×3, ReLU) → MaxPooling2D
  → Flatten
  → Dense(128, ReLU)
  → Dense(num_classes, Softmax)
```

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Image Size | 150 × 150 px |
| Epochs | 10 |
| Validation Split | 20% |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |

### Generated Graphs (after training)
1. 📈 **Accuracy** — Train vs Validation accuracy over epochs
2. 📉 **Loss** — Train vs Validation loss over epochs
3. 🎯 **Confusion Matrix** — True vs Predicted labels
4. 📊 **Per-Class Metrics** — Precision, Recall, F1 per class
5. 🏆 **Overall Performance** — Accuracy, Precision, Recall, F1
6. ✅ **Per-Class Accuracy** — Individual class accuracy bars

---

## 🔧 Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'flask'`
```bash
# Make sure your environment is activated
conda activate finalyearproject   # or: venv\Scripts\activate
pip install -r requirements.txt
```

### ❌ Model not found / "Train the model first"
- Go to `/train` and click **Start Training**

### ❌ Training finds 0 classes
- Ensure `train/` folder exists with subfolders containing images
- Each subfolder name = class name

### ❌ File upload rejected
- Allowed formats: PNG, JPG, JPEG, GIF, BMP
- Max size: 5MB

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
- The default key is for development only
- Never commit `.env` to version control
- Use HTTPS in production
- Consider a production WSGI server (Gunicorn) for deployment

---

## 📄 License

This project is for **educational and research purposes** only.
Always consult a qualified medical professional for cancer diagnosis.
