from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import seaborn as sns
from datetime import datetime
from dotenv import load_dotenv

from tensorflow.keras import layers, models

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global model cache
_model_cache = None
_training_in_progress = False

def discover_classes(train_path="train"):
    """Discover available class folders from the training directory"""
    if not os.path.exists(train_path):
        return []
    return sorted([d for d in os.listdir(train_path)
                   if os.path.isdir(os.path.join(train_path, d))])


# Suggestions mapped by normalized (lowercased, single-spaced) class name
_SUGGESTIONS = {
    'breast normal': "Healthy. Maintain regular screening.",
    'breast benign': "Non-cancerous tumor. Monitor regularly.",
    'breast malignant': "Cancer detected. Consult oncologist immediately.",
    'healthy liver': "Liver is healthy.",
    'liver hepatic_steatosis': "Fatty liver. Improve diet & lifestyle.",
    'skin dermatofibroma': "Benign skin condition.",
    'skin healthy': "Skin is healthy.",
    'skin seborrheic_keratosis': "Non-cancerous skin growth."
}


def get_suggestion(class_name):
    """Get medical suggestion using case/space-insensitive matching"""
    normalized = ' '.join(class_name.lower().split())
    return _SUGGESTIONS.get(normalized, "Consult doctor for further diagnosis.")


def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_status(
            id INTEGER PRIMARY KEY,
            is_trained INTEGER DEFAULT 0,
            trained_at TEXT,
            available_classes TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_model():
    """Load model once and cache it (singleton pattern)"""
    global _model_cache
    try:
        if _model_cache is None:
            if not os.path.exists("model.h5"):
                return None
            _model_cache = tf.keras.models.load_model("model.h5")
        return _model_cache
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def get_training_status():
    """Get training status from database"""
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT is_trained, available_classes FROM training_status WHERE id = 1")
        result = cursor.fetchone()
        conn.close()
        return result
    except Exception as e:
        print(f"Error getting training status: {str(e)}")
        return None

def set_training_status(is_trained, available_classes):
    """Update training status in database"""
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO training_status (id, is_trained, trained_at, available_classes) VALUES (1, ?, ?, ?)",
            (is_trained, datetime.now().isoformat(), ','.join(available_classes))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error setting training status: {str(e)}")


@app.route('/', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '').strip()

            if not username or not password:
                flash('Username and password are required', 'error')
                return render_template('signup.html')

            if len(password) < 6:
                flash('Password must be at least 6 characters', 'error')
                return render_template('signup.html')

            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM users WHERE username=?", (username,))
            if cursor.fetchone():
                flash('Username already exists', 'error')
                conn.close()
                return render_template('signup.html')

            cursor.execute("INSERT INTO users VALUES (NULL, ?, ?)",
                           (username, generate_password_hash(password)))
            conn.commit()
            conn.close()

            flash('Account created successfully! Please sign in.', 'success')
            return redirect(url_for('signin'))
        except Exception as e:
            flash(f'Error creating account: {str(e)}', 'error')
            return render_template('signup.html')

    return render_template('signup.html')


@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        try:
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '').strip()

            if not username or not password:
                flash('Username and password are required', 'error')
                return render_template('signin.html')

            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username=?", (username,))
            user = cursor.fetchone()
            conn.close()

            if user and check_password_hash(user[2], password):
                session['username'] = username
                flash('Welcome back!', 'success')
                return redirect(url_for('home'))
            else:
                flash('Invalid username or password', 'error')
        except Exception as e:
            flash(f'Error during login: {str(e)}', 'error')

    return render_template('signin.html')


@app.route('/home')
def home():
    if 'username' not in session:
        return redirect(url_for('signin'))
    return render_template('home.html')


@app.route('/detection', methods=['GET', 'POST'])
def detection():
    if 'username' not in session:
        return redirect(url_for('signin'))

    predicted_class = None
    suggestion = None
    image_path = None
    error_msg = None

    try:
        model = get_model()
        if model is None:
            error_msg = "Model not trained yet. Please train the model first."
            return render_template("detection.html",
                                   predicted_class=None,
                                   suggestion=None,
                                   image_path=None,
                                   error=error_msg)

        if request.method == 'POST':
            if 'image' not in request.files:
                error_msg = "No image file provided"
            else:
                file = request.files['image']

                if file.filename == '':
                    error_msg = "No file selected"
                elif not allowed_file(file.filename):
                    error_msg = f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
                else:
                    try:
                        # Create user-specific upload directory with timestamp
                        user_folder = os.path.join(UPLOAD_FOLDER, session['username'])
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        os.makedirs(user_folder, exist_ok=True)

                        filename = secure_filename(f"{timestamp}_{file.filename}")
                        image_path = os.path.join(user_folder, filename)
                        file.save(image_path)

                        img = image.load_img(image_path, target_size=(150, 150))
                        img = image.img_to_array(img) / 255.0
                        img = np.expand_dims(img, axis=0)

                        pred = model.predict(img)
                        label = np.argmax(pred)
                        confidence = float(np.max(pred)) * 100

                        # Get available classes from database
                        training_status = get_training_status()
                        if training_status and training_status[1]:
                            available_classes = training_status[1].split(',')
                        else:
                            available_classes = discover_classes()

                        if 0 <= label < len(available_classes):
                            predicted_class = available_classes[label]
                            suggestion = get_suggestion(predicted_class)
                        else:
                            error_msg = "Error: Invalid prediction index"

                        # Return relative path for frontend
                        image_path = image_path.replace('\\', '/').replace(UPLOAD_FOLDER, '/static/uploads')

                    except Exception as e:
                        error_msg = f"Error processing image: {str(e)}"

    except Exception as e:
        error_msg = f"Detection error: {str(e)}"

    return render_template("detection.html",
                           predicted_class=predicted_class,
                           suggestion=suggestion,
                           image_path=image_path,
                           error=error_msg,
                           confidence=locals().get('confidence'))



@app.route('/train')
def train():
    global _model_cache, _training_in_progress

    if _training_in_progress:
        return render_template("train.html", logs=["Training already in progress..."])

    logs = []
    force = request.args.get('force', '0') == '1'

    try:
        # Check if already trained (skip if force retrain)
        if not force:
            training_status = get_training_status()
            if training_status and training_status[0]:
                # Check if all graph files exist from previous training
                graphs_exist = all(os.path.exists(f"static/{g}") for g in [
                    'accuracy.png', 'loss.png', 'confusion_matrix.png',
                    'metrics.png', 'overall_stats.png', 'class_accuracy.png'
                ])
                return render_template("train.html",
                    logs=["✓ Model is already trained and ready for predictions."],
                    already_trained=True,
                    show_graphs=graphs_exist)

        # Clear stale cache and status when retraining
        _model_cache = None
        set_training_status(0, [])
        _training_in_progress = True

        IMAGE_SIZE = (150, 150)
        train_path = "train"
        test_path = "test"

        # Check paths exist
        if not os.path.exists(train_path):
            logs.append(f"ERROR: Training directory '{train_path}' not found")
            return render_template("train.html", logs=logs)

        if not os.path.exists(test_path):
            logs.append(f"ERROR: Test directory '{test_path}' not found")
            return render_template("train.html", logs=logs)

        # Dynamically discover classes from folder names on disk
        available_classes = discover_classes(train_path)

        logs.append(f"Classes Found ({len(available_classes)}): {available_classes}")

        if not available_classes:
            logs.append("ERROR: No class folders found in training directory!")
            return render_template("train.html", logs=logs)

        class_labels = {c: i for i, c in enumerate(available_classes)}

        def load_data(path):
            images, labels = [], []
            error_count = 0

            for category in available_classes:
                category_path = os.path.join(path, category)

                if not os.path.exists(category_path):
                    continue

                for file in os.listdir(category_path):
                    img_path = os.path.join(category_path, file)

                    try:
                        img = cv2.imread(img_path)

                        if img is None:
                            continue

                        img = cv2.resize(img, IMAGE_SIZE)
                        images.append(img)
                        labels.append(class_labels[category])

                    except Exception as e:
                        error_count += 1
                        continue

            if error_count > 0:
                logs.append(f"Warning: {error_count} images failed to load")

            return np.array(images) / 255.0, np.array(labels)

        logs.append("Loading training data...")
        train_images, train_labels = load_data(train_path)
        logs.append("Loading test data...")
        test_images, test_labels = load_data(test_path)

        logs.append(f"Train size: {len(train_images)}")
        logs.append(f"Test size: {len(test_images)}")

        if len(train_images) == 0:
            logs.append("ERROR: No training images found!")
            return render_template("train.html", logs=logs)

        logs.append("Building model...")
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            layers.MaxPooling2D(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(available_classes), activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        logs.append("Training model (10 epochs)...")
        history = model.fit(train_images, train_labels,
                            epochs=10,
                            validation_split=0.2,
                            verbose=0)

        logs.append("Saving model...")
        model.save("model.h5")
        with open("model.json", "w") as f:
            f.write(model.to_json())

        # Clear cached model so it reloads
        _model_cache = None

        # Get predictions
        preds = model.predict(test_images)
        y_pred = np.argmax(preds, axis=1)

        logs.append("Generating detailed training graphs...")

        # ===== GRAPH 1: ACCURACY PLOT =====
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], linewidth=2.5, label='Train Accuracy', marker='o')
        plt.plot(history.history['val_accuracy'], linewidth=2.5, label='Validation Accuracy', marker='s')
        plt.title("Model Accuracy Over Epochs", fontsize=14, fontweight='bold')
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("static/accuracy.png", dpi=100)
        plt.close()
        logs.append("✓ Graph 1: Accuracy plot saved")

        # ===== GRAPH 2: LOSS PLOT =====
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], linewidth=2.5, label='Train Loss', marker='o', color='#ff6b6b')
        plt.plot(history.history['val_loss'], linewidth=2.5, label='Validation Loss', marker='s', color='#ee5a6f')
        plt.title("Model Loss Over Epochs", fontsize=14, fontweight='bold')
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("static/loss.png", dpi=100)
        plt.close()
        logs.append("✓ Graph 2: Loss plot saved")

        # ===== GRAPH 3: CONFUSION MATRIX =====
        cm = confusion_matrix(test_labels, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=available_classes, yticklabels=available_classes, cbar_kws={'label': 'Count'})
        plt.title("Confusion Matrix - Detailed Predictions", fontsize=14, fontweight='bold')
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("static/confusion_matrix.png", dpi=100)
        plt.close()
        logs.append("✓ Graph 3: Confusion matrix saved")

        # ===== GRAPH 4: PER-CLASS METRICS =====
        precision_scores = precision_score(test_labels, y_pred, average=None, zero_division=0)
        recall_scores = recall_score(test_labels, y_pred, average=None, zero_division=0)
        f1_scores = f1_score(test_labels, y_pred, average=None, zero_division=0)

        x = np.arange(len(available_classes))
        width = 0.25

        plt.figure(figsize=(14, 6))
        plt.bar(x - width, precision_scores, width, label='Precision', alpha=0.8, color='#3b82f6')
        plt.bar(x, recall_scores, width, label='Recall', alpha=0.8, color='#10b981')
        plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8, color='#f59e0b')

        plt.title("Per-Class Performance Metrics", fontsize=14, fontweight='bold')
        plt.xlabel("Classes", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.xticks(x, available_classes, rotation=45, ha='right')
        plt.ylim([0, 1.1])
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig("static/metrics.png", dpi=100)
        plt.close()
        logs.append("✓ Graph 4: Per-class metrics saved")

        # ===== GRAPH 5: OVERALL STATISTICS =====
        overall_precision = precision_score(test_labels, y_pred, average='weighted', zero_division=0)
        overall_recall = recall_score(test_labels, y_pred, average='weighted', zero_division=0)
        overall_f1 = f1_score(test_labels, y_pred, average='weighted', zero_division=0)
        overall_accuracy = np.mean(y_pred == test_labels)

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [overall_accuracy, overall_precision, overall_recall, overall_f1]
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']

        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        plt.title("Overall Model Performance", fontsize=14, fontweight='bold')
        plt.ylabel("Score", fontsize=12)
        plt.ylim([0, 1.1])
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig("static/overall_stats.png", dpi=100)
        plt.close()
        logs.append("✓ Graph 5: Overall statistics saved")

        # ===== GRAPH 6: ACCURACY PER CLASS =====
        class_accuracies = []
        for i, cls in enumerate(available_classes):
            mask = test_labels == i
            if np.sum(mask) > 0:
                acc = np.mean(y_pred[mask] == test_labels[mask])
                class_accuracies.append(acc)
            else:
                class_accuracies.append(0)

        plt.figure(figsize=(12, 6))
        bars = plt.barh(available_classes, class_accuracies, color='#8b5cf6', alpha=0.7, edgecolor='black', linewidth=1.5)
        plt.title("Per-Class Accuracy", fontsize=14, fontweight='bold')
        plt.xlabel("Accuracy", fontsize=12)
        plt.xlim([0, 1.05])
        
        # Add percentage labels
        for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
            plt.text(acc + 0.02, i, f'{acc:.1%}', va='center', fontsize=10, fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig("static/class_accuracy.png", dpi=100)
        plt.close()
        logs.append("✓ Graph 6: Per-class accuracy saved")

        # Save training status to database
        set_training_status(1, available_classes)

        logs.append("✓ Model trained successfully!")

    except Exception as e:
        logs.append(f"ERROR during training: {str(e)}")
    finally:
        _training_in_progress = False

    return render_template("train.html", logs=logs, show_graphs=True)



@app.route('/model_info')
def model_info():
    if 'username' not in session:
        return redirect(url_for('signin'))

    info = {
        'model_exists': os.path.exists("model.h5") and os.path.exists("model.json"),
        'training_status': get_training_status()
    }
    return render_template('model_info.html', info=info)


@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('signin'))


@app.errorhandler(413)
def too_large(e):
    flash(f'File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f}MB', 'error')
    return redirect(url_for('detection'))


if __name__ == '__main__':
    app.run(debug=False)
