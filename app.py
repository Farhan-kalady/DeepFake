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
import matplotlib.cm as cm
import gc



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

def make_gradcam_heatmap(img_array, full_model, mobilenet_layer_name, last_conv_layer_name, pred_index=None):
    # Get the mobilenet layer
    mobilenet = full_model.get_layer(mobilenet_layer_name)
    
    # Create a model from mobilenet input to its target layer and final output
    grad_model = tf.keras.models.Model(
        mobilenet.inputs, 
        [mobilenet.get_layer(last_conv_layer_name).output, mobilenet.output]
    )

    # We need to run the data through the full model PREVIOUS to mobilenet if any
    # In this case mobilenet is the first meaningful layer after input
    
    with tf.GradientTape() as tape:
        # Run input through the grad_model
        # Note: img_array shape is (1, 128, 128, 3)
        last_conv_layer_output, mobilenet_output = grad_model(img_array)
        
        # Now pass mobilenet_output through the REST of the full_model
        # The full_model layers after mobilenet are: GlobalAveragePooling2D, Dropout, Dense
        x = mobilenet_output
        for layer_name in ['global_average_pooling2d', 'dropout', 'dense_1']:
            try:
                x = full_model.get_layer(layer_name)(x)
            except:
                # If layer names are different, we can also iterate by index
                continue
        
        preds = x
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.8):
    # Load the original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128)) # Resize to match model input for alignment
    img_float = img.astype("float32") / 255.0

    # Rescale heatmap to a range 0-255
    heatmap_uint8 = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap) / 255.0

    # For "Very Impressive" overlay, we want a high-contrast heatmap
    # We apply a threshold to the heatmap to keep only relevant hotspots
    # and set the rest to black so screen blend mode works perfectly
    heatmap_mask = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_mask = np.expand_dims(heatmap_mask, axis=-1)
    
    # Apply a soft threshold/contrast boost
    heatmap_mask = np.power(heatmap_mask, 1.5)
    
    # Final colorized heatmap on black background
    # This will look AMAZING with CSS mix-blend-mode: screen
    final_heatmap = jet_heatmap * heatmap_mask

    # Save the heatmap image
    final_heatmap_img = tf.keras.utils.array_to_img(final_heatmap)
    final_heatmap_img.save(cam_path)
    return cam_path



# 3. DETECTION LOGIC
def detect_media(file_path):
    ext = file_path.lower().split('.')[-1]
    
    # 3a. Disable Heatmap on low-RAM environments (Render Free Tier)
    # Set SKIP_HEATMAP=True in Render environment variables if it still crashes
    skip_heatmap = os.environ.get('SKIP_HEATMAP', 'False').lower() == 'true'
    is_render = 'RENDER' in os.environ
    
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
        
        
        # Calculate result and confidence before generating heatmap
        result = "FAKE" if avg_score > 0.5 else "REAL"
        confidence = round(float(avg_score if avg_score > 0.5 else 1 - avg_score) * 100, 2)
        
        # Generate Heatmap (Optional/Try-Except for RAM safety)
        heatmap_filename = None
        if not skip_heatmap and not is_render: # By default disable on Render to save RAM
            try:
                last_conv_layer_name = "out_relu"
                mobilenet_name = "mobilenetv2_1.00_128"
                heatmap = make_gradcam_heatmap(input_data, model, mobilenet_name, last_conv_layer_name)
                
                heatmap_filename = "heatmap_" + os.path.basename(file_path)
                heatmap_path = os.path.join(os.path.dirname(file_path), heatmap_filename)
                save_and_display_gradcam(file_path, heatmap, heatmap_path)
            except Exception as e:
                print(f"Heatmap generation failed (likely OOM): {e}")

        # Explicitly clear memory
        del input_data
        gc.collect()
        
        return result, confidence, heatmap_filename

    # Fallback for video or other cases
    result = "FAKE" if avg_score > 0.5 else "REAL"
    confidence = round(float(avg_score if avg_score > 0.5 else 1 - avg_score) * 100, 2)
    
    # Explicitly clear memory
    gc.collect()
    
    return result, confidence, None



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            result, confidence, heatmap_file = detect_media(file_path)
            
            # Relative path for HTML
            web_path = f"static/uploads/{filename}"
            heatmap_path = f"static/uploads/{heatmap_file}" if heatmap_file else None
            return render_template('index.html', result=result, confidence=confidence, file_path=web_path, heatmap_path=heatmap_path)

            
    return render_template('index.html')

if __name__ == '__main__':
    # Use the PORT environment variable if available, default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
