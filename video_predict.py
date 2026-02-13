import cv2
import numpy as np
import tensorflow as tf
import os

# 1. LOAD YOUR TRAINED MODEL
model = tf.keras.models.load_model('deepfake_model_v1.h5')

def predict_video(video_path, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    
    print(f"--- Starting Analysis for: {os.path.basename(video_path)} ---")

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every 'n-th' frame to save time and power
        if count % frame_interval == 0:
            # Preprocess: Resize to 128x128 and normalize to [0, 1]
            img = cv2.resize(frame, (128, 128))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict
            preds = model.predict(img, verbose=0)
            predictions.append(preds[0][0])

        count += 1

    cap.release()

    if not predictions:
        return "No frames captured. Check video path."

    # 2. CALCULATE FINAL RESULT
    # Average the scores of all sampled frames
    avg_score = np.mean(predictions)
    
    # Sigmoid output: > 0.5 is usually Fake
    result = "FAKE" if avg_score > 0.5 else "REAL"
    confidence = round(float(avg_score if avg_score > 0.5 else 1 - avg_score) * 100, 2)

    return f"RESULT: {result} | Confidence: {confidence}%"

# 3. RUN ON A TEST VIDEO
# Replace with a path from your 'Test' folder
test_video = r"E:\deep\Dataset\Test\your_video.mp4"
print(predict_video(test_video))