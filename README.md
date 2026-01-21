# face-recognition-attendance

"""
Face Recognition Attendance System
===================================

A complete web-based attendance system using CNN for face recognition.

Features:
---------
✅ CNN-based face recognition using TensorFlow
✅ OpenCV for face detection
✅ Admin panel for student registration
✅ Real-time face recognition and attendance marking
✅ Automatic attendance saving with timestamps
✅ Date-wise attendance viewing
✅ Modern, responsive UI with HTML/CSS

Project Structure:
------------------
face-attendance-system/
│
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
│
├── templates/            # HTML templates
│   ├── index.html        # Home page
│   ├── admin.html        # Admin panel
│   └── attendance.html   # Attendance page
│
├── students/             # Student data (created automatically)
│   └── [student_id]/
│       ├── info.json
│       └── face_0.jpg
│
├── attendance/           # Attendance records (created automatically)
│   └── [YYYY-MM-DD].json
│
├── face_recognition_model.h5      # Trained CNN model
└── label_encoder.pkl              # Label encoder

Installation Steps:
-------------------

1. Install Python 3.8 or higher

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install flask==2.3.3
   pip install opencv-python==4.8.0.76
   pip install numpy==1.24.3
   pip install tensorflow==2.13.0
   pip install scikit-learn==1.3.0
   ```

4. Create the project structure:
   - Create a folder named 'templates'
   - Save index.html, admin.html, and attendance.html in the templates folder
   - Save app.py in the root directory

5. Run the application:
   ```
   python app.py
   ```

6. Open your browser and navigate to:
   http://localhost:5000

Usage Guide:
------------

ADMIN PANEL (Register Students):
1. Go to Admin Panel
2. Enter Student ID (e.g., STU001)
3. Enter Student Name
4. Click "Start Camera"
5. Position face in camera view
6. Click "Capture Photo"
7. Click "Register Student"
8. Model will automatically retrain with new student

MARK ATTENDANCE:
1. Go to "View Attendance" page
2. Click "Start Camera"
3. Position face in camera view
4. Click "Mark Attendance"
5. System will recognize face and mark attendance
6. Attendance is saved automatically with timestamp

VIEW ATTENDANCE:
1. Select a date using the date picker
2. Click "Load Attendance"
3. View all students marked present on that date

Technical Details:
------------------

CNN Architecture:
- Conv2D (32 filters) → MaxPooling
- Conv2D (64 filters) → MaxPooling
- Conv2D (128 filters) → MaxPooling
- Flatten
- Dense (128) → Dropout (0.5)
- Dense (num_classes, softmax)

Face Detection:
- Haar Cascade Classifier (OpenCV)

Image Processing:
- Grayscale conversion
- Resize to 128x128 pixels
- Normalization (0-1 range)

Model Training:
- Automatic retraining when new student is registered
- 50 epochs with 20% validation split
- Adam optimizer
- Categorical cross-entropy loss

Attendance Storage:
- JSON format
- Organized by date (YYYY-MM-DD.json)
- Contains: student_id, name, time, status

Recognition Threshold:
- Confidence > 70% for positive identification

Tips for Best Results:
----------------------
1. Good lighting when capturing registration photos
2. Face should be clearly visible and centered
3. Neutral expression recommended
4. Register multiple photos per student for better accuracy
5. Ensure camera is at eye level during attendance marking

Troubleshooting:
----------------
- Camera not working: Check browser permissions
- Low accuracy: Register more photos per student
- Model not training: Ensure at least 2 students registered
- Face not detected: Improve lighting, position face clearly

Security Considerations:
------------------------
- Add authentication for admin panel in production
- Use HTTPS in production environment
- Implement rate limiting for API endpoints
- Add input validation and sanitization
- Store sensitive data securely

Future Enhancements:
--------------------
- Multi-face detection in single frame
- Export attendance to CSV/Excel
- Email notifications
- Mobile app integration
- Cloud storage for faces
- Anti-spoofing measures (liveness detection)

License: MIT
"""
