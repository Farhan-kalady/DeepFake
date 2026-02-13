import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' # Enable all logs
print("TensorFlow version:", tf.__version__)
try:
    print("Loading model...")
    model = tf.keras.models.load_model('deepfake_model_v1.h5')
    print("Model loaded successfully!")
    model.summary()
except Exception as e:
    print("Error loading model:", e)
