# 📐 Predictive Oncology — System Diagrams & Architecture

> **Project:** Deep Learning-Based Multi-Cancer Detection  
> **Stack:** Flask · TensorFlow/Keras · SQLite · Jinja2 · OpenCV · Matplotlib  
> **Generated:** April 2026

---

## Table of Contents

1. [Class Diagram](#1-class-diagram)
2. [Activity Diagram](#2-activity-diagram)
3. [End-to-End Flow Diagram](#3-end-to-end-flow-diagram)
4. [Complete Application Architecture](#4-complete-application-architecture)

---

## 1. Class Diagram

The application follows a **monolithic Flask architecture** where `app.py` serves as the single entry point. Although the codebase is not object-oriented in the traditional sense (no explicit Python classes beyond the Flask `app` object), it can be logically decomposed into the following conceptual classes and modules based on responsibility.

### 1.1 Class Diagram Description

```
┌─────────────────────────────────────────────────────────────────────┐
│                        «Flask Application»                          │
│                           FlaskApp                                  │
├─────────────────────────────────────────────────────────────────────┤
│ - app : Flask                                                       │
│ - secret_key : str                                                  │
│ - UPLOAD_FOLDER : str = "static/uploads"                            │
│ - ALLOWED_EXTENSIONS : set = {png, jpg, jpeg, gif, bmp}             │
│ - MAX_FILE_SIZE : int = 5MB                                         │
│ - _model_cache : tf.keras.Model = None                              │
│ - _training_in_progress : bool = False                              │
├─────────────────────────────────────────────────────────────────────┤
│ + signup() : Response           [GET/POST → /]                      │
│ + signin() : Response           [GET/POST → /signin]                │
│ + home() : Response             [GET → /home]                       │
│ + detection() : Response        [GET/POST → /detection]             │
│ + train() : Response            [GET → /train]                      │
│ + model_info() : Response       [GET → /model_info]                 │
│ + logout() : Response           [GET → /logout]                     │
│ + too_large(e) : Response       [Error Handler → 413]               │
└─────────────────────────────────────────────────────────────────────┘
         │ uses                │ uses                  │ uses
         ▼                    ▼                        ▼
┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────┐
│ «Utility Module» │  │  «Data Module»   │  │ «ML Pipeline Module»  │
│  Authentication  │  │  DatabaseManager │  │    ModelManager        │
├──────────────────┤  ├──────────────────┤  ├────────────────────────┤
│                  │  │ - DB : str =     │  │ - IMAGE_SIZE : tuple   │
│                  │  │   "users.db"     │  │   = (150, 150)         │
│                  │  │                  │  │ - train_path : str     │
│                  │  │                  │  │   = "train"            │
│                  │  │                  │  │ - test_path : str      │
│                  │  │                  │  │   = "test"             │
├──────────────────┤  ├──────────────────┤  ├────────────────────────┤
│ + allowed_file() │  │ + init_db()      │  │ + get_model()          │
│ + generate_      │  │ + get_training_  │  │ + discover_classes()   │
│   password_hash()│  │   status()       │  │ + load_data()          │
│ + check_password │  │ + set_training_  │  │ + build_cnn_model()    │
│   _hash()        │  │   status()       │  │ + train_model()        │
│ + secure_        │  │                  │  │ + predict()            │
│   filename()     │  │                  │  │ + generate_graphs()    │
│                  │  │                  │  │ + get_suggestion()     │
└──────────────────┘  └──────────────────┘  └────────────────────────┘
                              │                        │
                              ▼                        ▼
                    ┌──────────────────┐  ┌────────────────────────┐
                    │  «Persistence»   │  │  «Model Artifacts»    │
                    │   SQLite DB      │  │                        │
                    ├──────────────────┤  ├────────────────────────┤
                    │ Tables:          │  │ - model.h5             │
                    │ ┌──────────────┐ │  │   (Keras weights)      │
                    │ │ users        │ │  │ - model.json           │
                    │ │ ─ id (PK)    │ │  │   (Architecture JSON)  │
                    │ │ ─ username   │ │  │ - accuracy.png         │
                    │ │ ─ password   │ │  │ - loss.png             │
                    │ └──────────────┘ │  │ - confusion_matrix.png │
                    │ ┌──────────────┐ │  │ - metrics.png          │
                    │ │training_     │ │  │ - overall_stats.png    │
                    │ │  status      │ │  │ - class_accuracy.png   │
                    │ │ ─ id (PK)    │ │  └────────────────────────┘
                    │ │ ─ is_trained │ │
                    │ │ ─ trained_at │ │
                    │ │ ─ available_ │ │
                    │ │   classes    │ │
                    │ └──────────────┘ │
                    └──────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     «Presentation Layer»                            │
│                      Jinja2 Templates                               │
├─────────────────────────────────────────────────────────────────────┤
│ + signup.html      → User registration form with flash messages     │
│ + signin.html      → User login form with flash messages            │
│ + home.html        → Dashboard with feature cards & sidebar         │
│ + detection.html   → Image upload form + prediction results         │
│ + train.html       → Training logs + 6 performance graphs           │
│ + model_info.html  → Model status, architecture, and usage guide    │
└─────────────────────────────────────────────────────────────────────┘
         │ renders
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     «Static Assets»                                 │
│                      static/ directory                               │
├─────────────────────────────────────────────────────────────────────┤
│ + uploads/{username}/     → Per-user uploaded medical images         │
│ + accuracy.png            → Training accuracy graph (generated)      │
│ + loss.png                → Training loss graph (generated)          │
│ + confusion_matrix.png    → Confusion matrix heatmap (generated)     │
│ + metrics.png             → Per-class metrics chart (generated)      │
│ + overall_stats.png       → Overall performance bar chart (generated)│
│ + class_accuracy.png      → Per-class accuracy chart (generated)     │
│ + *.png / *.jpg           → Sample/reference images                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Relationships

| Relationship | Source | Target | Type | Description |
|---|---|---|---|---|
| **Uses** | FlaskApp (Routes) | Authentication Module | Dependency | Routes call `generate_password_hash`, `check_password_hash`, `allowed_file`, `secure_filename` for user & file security |
| **Uses** | FlaskApp (Routes) | DatabaseManager | Dependency | Routes call `init_db()`, `get_training_status()`, `set_training_status()` to persist user accounts and training metadata |
| **Uses** | FlaskApp (Routes) | ModelManager | Dependency | Routes call `get_model()`, `discover_classes()`, `get_suggestion()`, `load_data()` for ML operations |
| **Persists To** | DatabaseManager | SQLite (`users.db`) | Association | Two tables: `users` (id, username, password) and `training_status` (id, is_trained, trained_at, available_classes) |
| **Produces** | ModelManager | Model Artifacts | Association | Training generates `model.h5`, `model.json`, and 6 PNG graph files in `/static/` |
| **Renders** | FlaskApp (Routes) | Jinja2 Templates | Dependency | Each route renders a specific HTML template, passing context variables (predictions, logs, info, etc.) |
| **Serves** | Jinja2 Templates | Static Assets | Association | Templates embed generated graph images and user-uploaded images via `<img>` tags referencing `/static/` |

---

## 2. Activity Diagram

### 2.1 User Registration Activity

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │
                         ▼
               ┌─────────────────┐
               │  Visit / (root) │
               └────────┬────────┘
                         │
                         ▼
              ┌──────────────────┐
              │ Display Signup   │
              │ Form             │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ User Enters      │
              │ Username +       │
              │ Password         │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐        ┌──────────────────────┐
              │ Username empty   │──Yes──▶│ Flash "Username and  │
              │ OR Password      │        │ password required"   │
              │ empty?           │        │ → Re-render signup   │
              └────────┬─────────┘        └──────────────────────┘
                       │ No
                       ▼
              ┌──────────────────┐        ┌──────────────────────┐
              │ Password < 6     │──Yes──▶│ Flash "Password must │
              │ characters?      │        │ be >= 6 characters"  │
              └────────┬─────────┘        │ → Re-render signup   │
                       │ No               └──────────────────────┘
                       ▼
              ┌──────────────────┐        ┌──────────────────────┐
              │ Username already │──Yes──▶│ Flash "Username      │
              │ exists in DB?    │        │ already exists"      │
              └────────┬─────────┘        │ → Re-render signup   │
                       │ No               └──────────────────────┘
                       ▼
              ┌──────────────────┐
              │ Hash Password    │
              │ (Werkzeug bcrypt)│
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ INSERT into      │
              │ users table      │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Flash "Account   │
              │ created!"        │
              │ Redirect→/signin │
              └────────┬─────────┘
                       │
                       ▼
                  ┌─────────┐
                  │   END   │
                  └─────────┘
```

### 2.2 User Authentication Activity

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │
                         ▼
              ┌──────────────────┐
              │ Visit /signin    │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Display Login    │
              │ Form             │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ User Submits     │
              │ Credentials      │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐        ┌──────────────────────┐
              │ Fields empty?    │──Yes──▶│ Flash error          │
              └────────┬─────────┘        │ → Re-render signin   │
                       │ No               └──────────────────────┘
                       ▼
              ┌──────────────────┐
              │ Query DB for     │
              │ username         │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐        ┌──────────────────────┐
              │ User found AND   │──No───▶│ Flash "Invalid       │
              │ password hash    │        │ credentials"         │
              │ matches?         │        │ → Re-render signin   │
              └────────┬─────────┘        └──────────────────────┘
                       │ Yes
                       ▼
              ┌──────────────────┐
              │ Set session      │
              │ ['username']     │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Flash "Welcome!" │
              │ Redirect→/home   │
              └────────┬─────────┘
                       │
                       ▼
                  ┌─────────┐
                  │   END   │
                  └─────────┘
```

### 2.3 Model Training Activity

```
                       ┌─────────┐
                       │  START  │
                       └────┬────┘
                            │
                            ▼
                 ┌──────────────────┐
                 │ User navigates   │
                 │ to /train        │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐       ┌────────────────────────┐
                 │ Training already │──Yes─▶│ Return "Training       │
                 │ in progress?     │       │ already in progress"   │
                 └────────┬─────────┘       └────────────────────────┘
                          │ No
                          ▼
                 ┌──────────────────┐       ┌────────────────────────┐
                 │ Force retrain?   │──No──▶│ Check DB training      │
                 │ (?force=1)       │       │ status                 │
                 └────────┬─────────┘       └───────────┬────────────┘
                          │ Yes                         │
                          │                             ▼
                          │                  ┌──────────────────┐
                          │                  │ Already trained? │──Yes──▶ Show graphs
                          │                  └────────┬─────────┘         + "Retrain" btn
                          │                           │ No
                          ◀───────────────────────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Clear model      │
                 │ cache + Reset    │
                 │ training status  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐       ┌────────────────────────┐
                 │ Verify train/    │──No──▶│ Show error:            │
                 │ and test/ dirs   │       │ "Directory not found"  │
                 │ exist?           │       └────────────────────────┘
                 └────────┬─────────┘
                          │ Yes
                          ▼
                 ┌──────────────────┐
                 │ Discover classes │
                 │ from train/      │
                 │ subdirectories   │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐       ┌────────────────────────┐
                 │ Classes found?   │──No──▶│ Show error:            │
                 └────────┬─────────┘       │ "No class folders"     │
                          │ Yes             └────────────────────────┘
                          ▼
                 ┌──────────────────┐
                 │ Load & preprocess│
                 │ train images     │
                 │ (resize 150x150, │
                 │  normalize /255) │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Load & preprocess│
                 │ test images      │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Build CNN Model: │
                 │ Conv2D(32)→Pool  │
                 │ →Conv2D(64)→Pool │
                 │ →Flatten→Dense   │
                 │ (128)→Dense(N)   │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Compile (Adam,   │
                 │ sparse_cat_CE)   │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Train: 10 epochs │
                 │ validation=20%   │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Save model.h5    │
                 │ + model.json     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Generate 6       │
                 │ evaluation graphs│
                 │ (accuracy, loss, │
                 │  confusion, etc.)│
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Update DB:       │
                 │ is_trained = 1   │
                 │ + class list     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Render train.html│
                 │ with logs +      │
                 │ show_graphs=True │
                 └────────┬─────────┘
                          │
                          ▼
                     ┌─────────┐
                     │   END   │
                     └─────────┘
```

### 2.4 Cancer Detection / Prediction Activity

```
                       ┌─────────┐
                       │  START  │
                       └────┬────┘
                            │
                            ▼
                 ┌──────────────────┐       ┌──────────────────────┐
                 │ User logged in?  │──No──▶│ Redirect → /signin   │
                 └────────┬─────────┘       └──────────────────────┘
                          │ Yes
                          ▼
                 ┌──────────────────┐
                 │ Load cached model│
                 │ (singleton)      │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐       ┌──────────────────────┐
                 │ Model exists?    │──No──▶│ Show "Model not      │
                 └────────┬─────────┘       │ trained yet" error   │
                          │ Yes             └──────────────────────┘
                          ▼
                 ┌──────────────────┐
                 │ Display upload   │
                 │ form (GET)       │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ User uploads     │
                 │ image (POST)     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐       ┌──────────────────────┐
                 │ File present &   │──No──▶│ Show validation      │
                 │ extension valid? │       │ error message        │
                 └────────┬─────────┘       └──────────────────────┘
                          │ Yes
                          ▼
                 ┌──────────────────┐
                 │ Create user dir  │
                 │ (static/uploads/ │
                 │  {username}/)    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Save with secure │
                 │ timestamped name │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Preprocess:      │
                 │ • load_img       │
                 │   (150×150)      │
                 │ • img_to_array   │
                 │ • normalize /255 │
                 │ • expand_dims    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ model.predict()  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Get predicted    │
                 │ class index      │
                 │ (argmax) &       │
                 │ confidence (max) │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Map index →      │
                 │ class name from  │
                 │ available_classes│
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Look up medical  │
                 │ suggestion for   │
                 │ predicted class  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Render results:  │
                 │ • Uploaded image │
                 │ • Predicted class│
                 │ • Confidence %   │
                 │ • Medical advice │
                 └────────┬─────────┘
                          │
                          ▼
                     ┌─────────┐
                     │   END   │
                     └─────────┘
```

---

## 3. End-to-End Flow Diagram

This section describes the **complete user journey** from account creation to cancer detection, covering every interaction across the system.

### 3.1 Complete User Journey Flow

```
 ╔══════════════════════════════════════════════════════════════════════════════════╗
 ║                        END-TO-END USER FLOW                                     ║
 ╚══════════════════════════════════════════════════════════════════════════════════╝

 ┌───────────┐        ┌──────────────┐        ┌──────────────┐
 │   USER    │───────▶│   BROWSER    │───────▶│ FLASK SERVER │
 │ (Client)  │◀───────│  (Frontend)  │◀───────│  (Backend)   │
 └───────────┘        └──────────────┘        └──────┬───────┘
                                                     │
                                              ┌──────┴───────┐
                                              │              │
                                         ┌────▼────┐   ┌─────▼─────┐
                                         │ SQLite  │   │ TensorFlow│
                                         │   DB    │   │  Model    │
                                         └─────────┘   └───────────┘

 ═══════════════════════════════════════════════════════════════════════

 PHASE 1: ACCOUNT CREATION
 ══════════════════════════

   User                    Browser                     Flask Server              SQLite DB
    │                        │                              │                       │
    │──── Open Website ─────▶│                              │                       │
    │                        │──── GET / ──────────────────▶│                       │
    │                        │◀─── signup.html ────────────│                       │
    │◀─── Render Form ──────│                              │                       │
    │                        │                              │                       │
    │── Fill Username +     │                              │                       │
    │   Password ──────────▶│                              │                       │
    │                        │──── POST / ────────────────▶│                       │
    │                        │                              │── Validate inputs ──▶│
    │                        │                              │── Check duplicate ──▶│
    │                        │                              │◀── Not found ────────│
    │                        │                              │── Hash password ──── │
    │                        │                              │── INSERT user ──────▶│
    │                        │                              │◀── Success ──────────│
    │                        │◀── Redirect /signin ────────│                       │
    │◀── Flash "Created!" ──│                              │                       │

 PHASE 2: AUTHENTICATION
 ════════════════════════

    │                        │                              │                       │
    │── Enter Credentials ──▶│                              │                       │
    │                        │──── POST /signin ──────────▶│                       │
    │                        │                              │── Query username ───▶│
    │                        │                              │◀── Return user row ──│
    │                        │                              │── Verify hash ─────  │
    │                        │                              │── Set session ─────  │
    │                        │◀── Redirect /home ──────────│                       │
    │◀── Dashboard ─────────│                              │                       │

 PHASE 3: MODEL TRAINING (one-time / on-demand)
 ═══════════════════════════════════════════════

    │                        │                              │              TensorFlow
    │── Click "Train" ──────▶│                              │                  │
    │                        │── GET /train ──────────────▶│                  │
    │                        │                              │── Check status ─▶│ (DB)
    │                        │                              │                  │
    │                        │                              │── Discover class │
    │                        │                              │   folders (train/)│
    │                        │                              │                  │
    │                        │                              │── Load images ── │
    │                        │                              │   (OpenCV)       │
    │                        │                              │── Resize 150×150 │
    │                        │                              │── Normalize /255 │
    │                        │                              │                  │
    │                        │                              │── Build CNN ────▶│
    │                        │                              │── Compile ──────▶│
    │                        │                              │── Fit 10 epochs ▶│
    │                        │                              │◀── History ──────│
    │                        │                              │                  │
    │                        │                              │── Save model.h5 ─│
    │                        │                              │── Save model.json│
    │                        │                              │                  │
    │                        │                              │── Generate 6 ────│
    │                        │                              │   PNG graphs     │
    │                        │                              │   (matplotlib)   │
    │                        │                              │                  │
    │                        │                              │── Update DB ────▶│ (DB)
    │                        │◀── train.html + graphs ─────│                  │
    │◀── View Results ──────│                              │                  │

 PHASE 4: CANCER DETECTION (core feature)
 ════════════════════════════════════════

    │                        │                              │              TensorFlow
    │── Click "Detection" ──▶│                              │                  │
    │                        │── GET /detection ──────────▶│                  │
    │                        │◀── Upload form ─────────────│                  │
    │                        │                              │                  │
    │── Select Image File ──▶│                              │                  │
    │── Click "Analyze" ────▶│                              │                  │
    │                        │── POST /detection ─────────▶│                  │
    │                        │   (multipart/form-data)     │                  │
    │                        │                              │── Validate file ──│
    │                        │                              │── Save to uploads │
    │                        │                              │── Preprocess ─────│
    │                        │                              │   (load, resize,  │
    │                        │                              │    normalize,     │
    │                        │                              │    expand_dims)   │
    │                        │                              │                  │
    │                        │                              │── model.predict()▶│
    │                        │                              │◀── probabilities ─│
    │                        │                              │                  │
    │                        │                              │── argmax → class  │
    │                        │                              │── max → confidence│
    │                        │                              │── Map → suggestion│
    │                        │                              │                  │
    │                        │◀── detection.html ──────────│                  │
    │                        │    (class, confidence,       │                  │
    │                        │     suggestion, image)       │                  │
    │◀── View Prediction ───│                              │                  │

 PHASE 5: SESSION END
 ════════════════════

    │── Click "Logout" ─────▶│                              │
    │                        │── GET /logout ──────────────▶│
    │                        │                              │── Clear session ──│
    │                        │◀── Redirect /signin ────────│
    │◀── Login Page ────────│                              │
```

### 3.2 Data Flow Summary Table

| Step | Input | Process | Output | Storage |
|---|---|---|---|---|
| **1. Signup** | Username + Password | Validate → Hash → Store | User record | `users` table (SQLite) |
| **2. Signin** | Username + Password | Query → Verify Hash → Session | Session cookie | Flask session (server-side) |
| **3. Train** | `train/` + `test/` dirs | Load → Preprocess → Build CNN → Fit → Evaluate | model.h5, 6 PNG graphs | Filesystem + `training_status` table |
| **4. Detect** | Medical image (upload) | Validate → Save → Preprocess → Predict → Map class | Predicted class + confidence + suggestion | `uploads/{user}/` dir |
| **5. Logout** | Session token | Clear session | Redirect to signin | Session cleared |

### 3.3 Request-Response Mapping

| Route | Method | Auth Required | Request Data | Response |
|---|---|---|---|---|
| `/` | GET | ❌ | — | `signup.html` |
| `/` | POST | ❌ | `username`, `password` | Redirect → `/signin` or re-render with error |
| `/signin` | GET | ❌ | — | `signin.html` |
| `/signin` | POST | ❌ | `username`, `password` | Redirect → `/home` or re-render with error |
| `/home` | GET | ✅ | — | `home.html` (dashboard) |
| `/detection` | GET | ✅ | — | `detection.html` (upload form) |
| `/detection` | POST | ✅ | `image` (file) | `detection.html` (with prediction results) |
| `/train` | GET | ❌ | `?force=1` (optional) | `train.html` (logs + graphs) |
| `/model_info` | GET | ✅ | — | `model_info.html` (model details) |
| `/logout` | GET | ✅ | — | Redirect → `/signin` |

---

## 4. Complete Application Architecture

### 4.1 High-Level Architecture Diagram

```
 ╔═══════════════════════════════════════════════════════════════════════════════╗
 ║                    PREDICTIVE ONCOLOGY — SYSTEM ARCHITECTURE                  ║
 ╠═══════════════════════════════════════════════════════════════════════════════╣
 ║                                                                               ║
 ║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
 ║  │                          CLIENT LAYER                                   │  ║
 ║  │                                                                         │  ║
 ║  │   ┌──────────┐    ┌──────────────────────────────────────────────┐      │  ║
 ║  │   │  Web     │    │         Jinja2 HTML Templates                │      │  ║
 ║  │   │ Browser  │◀──▶│  signup · signin · home · detection          │      │  ║
 ║  │   │ (Client) │    │  train  · model_info                         │      │  ║
 ║  │   └──────────┘    │                                              │      │  ║
 ║  │                   │  CSS: Poppins Font, Glassmorphism, Gradients  │      │  ║
 ║  │                   │  JS:  Jinja2 template variables only         │      │  ║
 ║  │                   └──────────────────────────────────────────────┘      │  ║
 ║  └─────────────────────────────────────┬───────────────────────────────────┘  ║
 ║                                        │ HTTP (Port 5000)                     ║
 ║                                        ▼                                      ║
 ║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
 ║  │                        APPLICATION LAYER                                │  ║
 ║  │                                                                         │  ║
 ║  │   ┌─────────────────────────────────────────────────────────────────┐   │  ║
 ║  │   │                    Flask Web Server (app.py)                     │   │  ║
 ║  │   │                    Werkzeug WSGI / Dev Server                    │   │  ║
 ║  │   └────────────────────────────┬────────────────────────────────────┘   │  ║
 ║  │                                │                                        │  ║
 ║  │    ┌───────────────────────────┼───────────────────────────────┐        │  ║
 ║  │    │                           │                               │        │  ║
 ║  │    ▼                           ▼                               ▼        │  ║
 ║  │  ┌──────────────┐  ┌────────────────────┐  ┌──────────────────────┐    │  ║
 ║  │  │ AUTH MODULE  │  │  TRAINING MODULE   │  │  DETECTION MODULE   │    │  ║
 ║  │  │              │  │                    │  │                      │    │  ║
 ║  │  │ • signup()   │  │ • train()          │  │ • detection()        │    │  ║
 ║  │  │ • signin()   │  │ • discover_        │  │ • get_model()        │    │  ║
 ║  │  │ • logout()   │  │   classes()        │  │ • allowed_file()     │    │  ║
 ║  │  │ • session    │  │ • load_data()      │  │ • get_suggestion()   │    │  ║
 ║  │  │   management │  │ • build_model()    │  │ • image preprocess   │    │  ║
 ║  │  │              │  │ • generate_graphs()│  │ • model.predict()    │    │  ║
 ║  │  └──────┬───────┘  └─────────┬──────────┘  └──────────┬───────────┘    │  ║
 ║  │         │                    │                         │                │  ║
 ║  └─────────┼────────────────────┼─────────────────────────┼────────────────┘  ║
 ║            │                    │                         │                    ║
 ║            ▼                    ▼                         ▼                    ║
 ║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
 ║  │                          DATA LAYER                                     │  ║
 ║  │                                                                         │  ║
 ║  │  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────────┐ │  ║
 ║  │  │  SQLite Database  │  │  Filesystem      │  │  ML Model Storage    │ │  ║
 ║  │  │  (users.db)       │  │  (static/)       │  │                      │ │  ║
 ║  │  │                   │  │                   │  │  • model.h5          │ │  ║
 ║  │  │  ┌─────────────┐ │  │  • uploads/       │  │    (Keras weights)   │ │  ║
 ║  │  │  │ users       │ │  │    {user}/        │  │                      │ │  ║
 ║  │  │  │ • id (PK)   │ │  │    {timestamp}_   │  │  • model.json        │ │  ║
 ║  │  │  │ • username  │ │  │    {filename}      │  │    (Architecture)    │ │  ║
 ║  │  │  │ • password  │ │  │                   │  │                      │ │  ║
 ║  │  │  │   (hashed)  │ │  │  • accuracy.png   │  └───────────────────────┘ │  ║
 ║  │  │  └─────────────┘ │  │  • loss.png       │                            │  ║
 ║  │  │  ┌─────────────┐ │  │  • confusion_     │                            │  ║
 ║  │  │  │ training_   │ │  │    matrix.png     │                            │  ║
 ║  │  │  │ status      │ │  │  • metrics.png    │                            │  ║
 ║  │  │  │ • id (PK)   │ │  │  • overall_       │                            │  ║
 ║  │  │  │ • is_trained│ │  │    stats.png      │                            │  ║
 ║  │  │  │ • trained_at│ │  │  • class_         │                            │  ║
 ║  │  │  │ • available_│ │  │    accuracy.png   │                            │  ║
 ║  │  │  │   classes   │ │  │                   │                            │  ║
 ║  │  │  └─────────────┘ │  └──────────────────┘                             │  ║
 ║  │  └──────────────────┘                                                    │  ║
 ║  │                                                                         │  ║
 ║  └─────────────────────────────────────────────────────────────────────────┘  ║
 ║                                                                               ║
 ║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
 ║  │                     ML / DEEP LEARNING LAYER                            │  ║
 ║  │                                                                         │  ║
 ║  │  ┌──────────────────────────────────────────────────────────────────┐   │  ║
 ║  │  │                  CNN Architecture                                │   │  ║
 ║  │  │                                                                  │   │  ║
 ║  │  │  Input(150×150×3)                                                │   │  ║
 ║  │  │    │                                                             │   │  ║
 ║  │  │    ▼                                                             │   │  ║
 ║  │  │  Conv2D(32, 3×3, ReLU) ──▶ MaxPooling2D(2×2)                    │   │  ║
 ║  │  │    │                                                             │   │  ║
 ║  │  │    ▼                                                             │   │  ║
 ║  │  │  Conv2D(64, 3×3, ReLU) ──▶ MaxPooling2D(2×2)                    │   │  ║
 ║  │  │    │                                                             │   │  ║
 ║  │  │    ▼                                                             │   │  ║
 ║  │  │  Flatten ──▶ Dense(128, ReLU) ──▶ Dense(8, Softmax)              │   │  ║
 ║  │  │                                       │                          │   │  ║
 ║  │  │                                       ▼                          │   │  ║
 ║  │  │                          Probability Distribution                │   │  ║
 ║  │  │                          (8 cancer classes)                       │   │  ║
 ║  │  └──────────────────────────────────────────────────────────────────┘   │  ║
 ║  │                                                                         │  ║
 ║  │  Training Config:  Adam | SparseCatCE | 10 epochs | 20% val split      │  ║
 ║  │  Evaluation:       Accuracy, Precision, Recall, F1, Confusion Matrix   │  ║
 ║  │  Libraries:        TensorFlow 2.13 | OpenCV 4.8 | scikit-learn 1.3     │  ║
 ║  └─────────────────────────────────────────────────────────────────────────┘  ║
 ║                                                                               ║
 ║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
 ║  │                     DEPLOYMENT LAYER (Optional)                         │  ║
 ║  │                                                                         │  ║
 ║  │   Dockerfile ─────▶  Python 3.10-slim + OpenCV system deps              │  ║
 ║  │   docker-compose ──▶  Port 5000 | Volumes: train/, test/, uploads       │  ║
 ║  │   .env.example ────▶  SECRET_KEY, FLASK_ENV configuration               │  ║
 ║  └─────────────────────────────────────────────────────────────────────────┘  ║
 ╚═══════════════════════════════════════════════════════════════════════════════╝
```

### 4.2 Technology Stack Breakdown

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Web Framework** | Flask | 2.3.3 | HTTP routing, session management, template rendering |
| **Template Engine** | Jinja2 | (bundled with Flask) | Server-side HTML rendering with dynamic data |
| **Security** | Werkzeug | 2.3.7 | Password hashing (`generate_password_hash`, `check_password_hash`), secure file handling |
| **Deep Learning** | TensorFlow / Keras | 2.13.0 | CNN model definition, training, prediction (`Sequential`, `Conv2D`, `Dense`) |
| **Image Processing** | OpenCV | 4.8.0 | Image reading (`cv2.imread`), resizing (`cv2.resize`) for training data |
| **Image Loading** | Keras Preprocessing | (part of TF) | Image loading for prediction (`image.load_img`, `image.img_to_array`) |
| **Scientific Computing** | NumPy | 1.24.3 | Array operations, normalization, argmax for predictions |
| **Visualization** | Matplotlib | 3.7.2 | Generating 6 training evaluation plots (accuracy, loss, confusion, etc.) |
| **Visualization** | Seaborn | 0.12.2 | Confusion matrix heatmap rendering |
| **ML Metrics** | scikit-learn | 1.3.0 | `confusion_matrix`, `precision_score`, `recall_score`, `f1_score` |
| **Database** | SQLite3 | (stdlib) | User accounts and training metadata storage |
| **Config** | python-dotenv | 1.0.0 | Environment variable loading from `.env` file |
| **Containerization** | Docker | — | Dockerfile + docker-compose for isolated deployment |

### 4.3 Directory Architecture Map

```
Predictive Oncology Deep Learning-Based Multi-Cancer Detection/
│
├── app.py                          ← Core application: all routes, ML pipeline, DB ops
├── model.h5                        ← Saved Keras model weights (generated after training)
├── users.db                        ← SQLite database (auto-created)
│
├── requirements.txt                ← Python dependencies (9 packages)
├── .env.example                    ← Environment config template
├── .gitignore                      ← Git exclusions
├── .dockerignore                   ← Docker build exclusions
├── Dockerfile                      ← Container image definition
├── docker-compose.yml              ← Multi-service orchestration
├── README.md                       ← Project documentation
│
├── templates/                      ← Jinja2 HTML templates (6 pages)
│   ├── signup.html                 ← Registration page with validation
│   ├── signin.html                 ← Login page with session management
│   ├── home.html                   ← Dashboard with feature cards
│   ├── detection.html              ← Image upload + prediction results
│   ├── train.html                  ← Training logs + 6 performance graphs
│   └── model_info.html             ← Model architecture + status info
│
├── static/                         ← Static assets served by Flask
│   ├── uploads/                    ← Per-user uploaded images
│   │   └── {username}/             ← User-specific directory
│   │       └── {timestamp}_{file}  ← Timestamped secure filenames
│   ├── accuracy.png                ← Generated: accuracy over epochs
│   ├── loss.png                    ← Generated: loss over epochs
│   ├── confusion_matrix.png        ← Generated: confusion matrix heatmap
│   ├── metrics.png                 ← Generated: per-class precision/recall/F1
│   ├── overall_stats.png           ← Generated: overall performance bars
│   └── class_accuracy.png          ← Generated: per-class accuracy bars
│
├── train/                          ← Training dataset (8 class folders)
│   ├── Breast  Normal/             ← ~187 images
│   ├── Breast Benign/              ← ~204 images
│   ├── Breast Malignant/           ← ~200 images
│   ├── Healthy Liver/              ← ~200 images
│   ├── Liver Hepatic_Steatosis/    ← ~191 images
│   ├── Skin Dermatofibroma/        ← ~95 images
│   ├── Skin Healthy/               ← ~143 images
│   └── Skin seborrheic_keratosis/  ← ~18 images
│
└── test/                           ← Test dataset (same 8 class folders)
    ├── Breast  Normal/
    ├── Breast Benign/
    ├── Breast Malignant/
    ├── Healthy Liver/
    ├── Liver Hepatic_Steatosis/
    ├── Skin Dermatofibroma/
    ├── Skin Healthy/
    └── Skin seborrheic_keratosis/
```

### 4.4 Security Architecture

| Concern | Implementation | Details |
|---|---|---|
| **Password Storage** | Werkzeug `generate_password_hash` | Bcrypt-based salted hash — plaintext passwords are NEVER stored |
| **Session Management** | Flask `session` with `secret_key` | Server-side signed cookies; configured via `.env` |
| **Route Protection** | Manual `session['username']` check | Protected routes (`/home`, `/detection`, `/model_info`, `/logout`) redirect to `/signin` if unauthenticated |
| **File Upload Security** | `secure_filename()` + extension whitelist | Only `png, jpg, jpeg, gif, bmp` allowed; max 5MB; timestamped names prevent collisions |
| **SQL Injection Prevention** | Parameterized queries (`?` placeholders) | All database queries use parameterized statements |
| **Error Handling** | Try-catch blocks + flash messages | Errors are caught and displayed as user-friendly messages; no stack traces exposed |

### 4.5 Cancer Classification Mapping

The model classifies images into **8 classes** across **3 organ types**:

| # | Class Name | Organ | Severity | Medical Suggestion |
|---|---|---|---|---|
| 0 | Breast Normal | Breast | ✅ Healthy | "Healthy. Maintain regular screening." |
| 1 | Breast Benign | Breast | ⚠️ Monitor | "Non-cancerous tumor. Monitor regularly." |
| 2 | Breast Malignant | Breast | 🔴 Critical | "Cancer detected. Consult oncologist immediately." |
| 3 | Healthy Liver | Liver | ✅ Healthy | "Liver is healthy." |
| 4 | Liver Hepatic_Steatosis | Liver | ⚠️ Monitor | "Fatty liver. Improve diet & lifestyle." |
| 5 | Skin Dermatofibroma | Skin | ⚠️ Benign | "Benign skin condition." |
| 6 | Skin Healthy | Skin | ✅ Healthy | "Skin is healthy." |
| 7 | Skin seborrheic_keratosis | Skin | ⚠️ Benign | "Non-cancerous skin growth." |

### 4.6 Model Training Pipeline Detail

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                    MODEL TRAINING PIPELINE                               │
 │                                                                          │
 │  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌──────────────┐ │
 │  │  Dataset    │    │  Loading   │    │ Preprocess │    │   Training   │ │
 │  │  Discovery  │───▶│  (OpenCV)  │───▶│            │───▶│   (Keras)    │ │
 │  │            │    │            │    │            │    │              │ │
 │  │ Scan train/│    │ cv2.imread │    │ resize     │    │ model.fit()  │ │
 │  │ subfolders │    │ per class  │    │ 150×150    │    │ 10 epochs    │ │
 │  │ → 8 classes│    │ folder     │    │ /255.0     │    │ 20% val      │ │
 │  └────────────┘    └────────────┘    └────────────┘    └──────┬───────┘ │
 │                                                               │         │
 │                                                               ▼         │
 │  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌──────────────┐ │
 │  │  Persist   │    │ Viz: 6     │    │ Metrics    │    │  Evaluate    │ │
 │  │            │◀───│ PNG Graphs │◀───│ Compute    │◀───│  on test set │ │
 │  │            │    │            │    │            │    │              │ │
 │  │ model.h5   │    │ matplotlib │    │ precision  │    │ model.       │ │
 │  │ model.json │    │ + seaborn  │    │ recall     │    │ predict()    │ │
 │  │ DB status  │    │            │    │ f1, confmx │    │ → y_pred     │ │
 │  └────────────┘    └────────────┘    └────────────┘    └──────────────┘ │
 └──────────────────────────────────────────────────────────────────────────┘
```

### 4.7 Caching & Performance Strategy

| Strategy | Implementation | Benefit |
|---|---|---|
| **Model Singleton** | `_model_cache` global variable — `get_model()` loads once from `model.h5`, caches in memory | Avoids re-loading a ~127MB model file on every prediction request |
| **Training Lock** | `_training_in_progress` boolean flag | Prevents concurrent training sessions from corrupting the model |
| **Graph Cache-Busting** | `?t={{ range(1,1000000)\|random }}` query param on image URLs | Forces browser to fetch fresh graphs after retraining |
| **User Upload Isolation** | `static/uploads/{username}/` per-user directories | Prevents filename collisions between users |
| **Timestamped Filenames** | `{YYYYMMDD_HHMMSS}_{filename}` format | Prevents same-user filename collisions across multiple uploads |

---

> **Disclaimer:** This system is for educational and research purposes only. Do not rely on AI predictions for actual medical diagnoses. Always consult qualified healthcare professionals.
