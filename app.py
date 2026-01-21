from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
import os
from datetime import datetime
import json
import base64
import traceback

# TensorFlow imports
try:
    from tensorflow import keras
    from keras.models import Sequential, load_model
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from keras.utils import to_categorical
except ImportError:
    # Fallback for older TensorFlow versions
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)
CORS(app)

# Configuration
STUDENTS_DIR = 'students'
ATTENDANCE_DIR = 'attendance'
MODEL_PATH = 'face_recognition_model.h5'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
IMG_SIZE = 128
IMAGES_PER_STUDENT = 5  # Capture multiple images per student

# Create necessary directories
os.makedirs(STUDENTS_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Global variables
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = None
label_encoder = None

# CNN Model Architecture
def create_cnn_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Load or initialize model
def load_or_create_model():
    global model, label_encoder
    
    if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
        model = load_model(MODEL_PATH)
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        return True
    return False

# Train model with registered students
def train_model():
    global model, label_encoder
    
    images = []
    labels = []
    
    # Load all student images
    for student_id in os.listdir(STUDENTS_DIR):
        student_path = os.path.join(STUDENTS_DIR, student_id)
        if os.path.isdir(student_path):
            for img_name in os.listdir(student_path):
                if img_name.endswith(('.jpg', '.png')):
                    img_path = os.path.join(student_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        # Apply histogram equalization for better contrast
                        img = cv2.equalizeHist(img)
                        images.append(img)
                        labels.append(student_id)
    
    if len(images) == 0:
        return False
    
    # Prepare data
    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)
    
    # Create and train model
    num_classes = len(label_encoder.classes_)
    model = create_cnn_model(num_classes)
    
    # Increased epochs and added early stopping patience
    model.fit(images, labels_categorical, epochs=100, batch_size=16, validation_split=0.2, verbose=1)
    
    # Save model and encoder
    model.save(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    return True

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/attendance')
def attendance_page():
    return render_template('attendance.html')

@app.route('/api/register_student', methods=['POST'])
def register_student():
    try:
        data = request.json
        student_id = data.get('student_id')
        student_name = data.get('student_name')
        image_data = data.get('image')
        
        print(f"Received registration request for: {student_id} - {student_name}")
        
        if not all([student_id, student_name, image_data]):
            return jsonify({'success': False, 'message': 'Missing data'}), 400
        
        # Create student directory
        student_dir = os.path.join(STUDENTS_DIR, student_id)
        os.makedirs(student_dir, exist_ok=True)
        
        # Save student info
        info_path = os.path.join(student_dir, 'info.json')
        with open(info_path, 'w') as f:
            json.dump({'id': student_id, 'name': student_name}, f)
        
        # Decode and save image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'success': False, 'message': 'Invalid image data'}), 400
        
        # Detect face and save
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'No face detected in image'}), 400
        
        # Save the largest face (most prominent)
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]
        
        # Add padding around face
        padding = int(w * 0.2)
        y1 = max(0, y - padding)
        y2 = min(gray.shape[0], y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(gray.shape[1], x + w + padding)
        
        face = gray[y1:y2, x1:x2]
        face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        
        # Save original image
        cv2.imwrite(os.path.join(student_dir, f'face_0.jpg'), face_resized)
        
        print(f"Student {student_id} registered successfully")
        
        # Retrain model
        training_success = train_model()
        if training_success:
            load_or_create_model()
            return jsonify({'success': True, 'message': 'Student registered and model trained successfully'})
        else:
            return jsonify({'success': True, 'message': 'Student registered. Model will train after more students are added.'})
    
    except Exception as e:
        print(f"Error in register_student: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500

@app.route('/api/get_students', methods=['GET'])
def get_students():
    students = []
    for student_id in os.listdir(STUDENTS_DIR):
        info_path = os.path.join(STUDENTS_DIR, student_id, 'info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                students.append(json.load(f))
    return jsonify({'students': students})

@app.route('/api/mark_attendance', methods=['POST'])
def mark_attendance():
    try:
        global model, label_encoder
        
        if model is None:
            load_or_create_model()
        
        if model is None:
            return jsonify({'success': False, 'message': 'No trained model available. Please register students first.'}), 400
        
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image data'}), 400
        
        # Decode image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization
        gray = cv2.equalizeHist(gray)
        
        # Detect faces with better parameters
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'No face detected'}), 400
        
        recognized_students = []
        
        # Process largest face
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]
        
        # Add padding
        padding = int(w * 0.2)
        y1 = max(0, y - padding)
        y2 = min(gray.shape[0], y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(gray.shape[1], x + w + padding)
        
        face = gray[y1:y2, x1:x2]
        face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face_normalized = face_resized.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
        
        # Predict
        predictions = model.predict(face_normalized, verbose=0)
        confidence = np.max(predictions)
        
        print(f"Predictions: {predictions}")
        print(f"Confidence: {confidence}")
        
        if confidence > 0.5:  # Lowered threshold from 0.7 to 0.5
                predicted_label = np.argmax(predictions)
                student_id = label_encoder.inverse_transform([predicted_label])[0]
                
                # Get student info
                info_path = os.path.join(STUDENTS_DIR, student_id, 'info.json')
                with open(info_path, 'r') as f:
                    student_info = json.load(f)
                
                # Mark attendance
                today = datetime.now().strftime('%Y-%m-%d')
                time_now = datetime.now().strftime('%H:%M:%S')
                
                attendance_file = os.path.join(ATTENDANCE_DIR, f'{today}.json')
                attendance_data = {}
                
                if os.path.exists(attendance_file):
                    with open(attendance_file, 'r') as f:
                        attendance_data = json.load(f)
                
                if student_id not in attendance_data:
                    attendance_data[student_id] = {
                        'name': student_info['name'],
                        'time': time_now,
                        'status': 'Present'
                    }
                    
                    with open(attendance_file, 'w') as f:
                        json.dump(attendance_data, f, indent=2)
                    
                    recognized_students.append({
                        'id': student_id,
                        'name': student_info['name'],
                        'confidence': float(confidence)
                    })
        
        if recognized_students:
            return jsonify({
                'success': True,
                'message': 'Attendance marked successfully',
                'students': recognized_students
            })
        else:
            return jsonify({'success': False, 'message': 'Face not recognized or already marked'}), 400
    
    except Exception as e:
        print(f"Error in mark_attendance: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500

@app.route('/api/get_attendance', methods=['GET'])
def get_attendance():
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    attendance_file = os.path.join(ATTENDANCE_DIR, f'{date}.json')
    
    if os.path.exists(attendance_file):
        with open(attendance_file, 'r') as f:
            attendance_data = json.load(f)
        return jsonify({'success': True, 'attendance': attendance_data, 'date': date})
    else:
        return jsonify({'success': True, 'attendance': {}, 'date': date})

if __name__ == '__main__':
    load_or_create_model()
    app.run(debug=True, host='0.0.0.0', port=5000)