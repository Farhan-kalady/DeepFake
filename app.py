import os
# Silences oneDNN and CPU instruction logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
# Further silences ABSL warnings
import logging
tf.get_logger().setLevel('ERROR')
import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 1. LOAD MODEL
# Load your model (ensure this file is in E:\deep\)
model = tf.keras.models.load_model('deepfake_model_v1.h5')

# 2. CONFIGURATION
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper: Preprocess single frame
def preprocess_frame(frame):
    img = cv2.resize(frame, (128, 128))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# 3. DETECTION LOGIC
def detect_media(file_path):
    ext = file_path.lower().split('.')[-1]
    
    # VIDEO LOGIC
    if ext in ['mp4', 'avi', 'mov']:
        cap = cv2.VideoCapture(file_path)
        scores = []
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if count % 10 == 0: # Process every 10th frame
                input_data = preprocess_frame(frame)
                pred = model.predict(input_data, verbose=0)[0][0]
                scores.append(pred)
            count += 1
        cap.release()
        avg_score = np.mean(scores) if scores else 0
        
    # IMAGE LOGIC
    else:
        img = cv2.imread(file_path)
        input_data = preprocess_frame(img)
        avg_score = model.predict(input_data, verbose=0)[0][0]

    result = "FAKE" if avg_score > 0.5 else "REAL"
    confidence = round(float(avg_score if avg_score > 0.5 else 1 - avg_score) * 100, 2)
    return result, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            result, confidence = detect_media(file_path)
            
            # Relative path for HTML
            web_path = f"static/uploads/{filename}"
            return render_template('index.html', result=result, confidence=confidence, file_path=web_path)
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)