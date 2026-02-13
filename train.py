import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models, applications

# --- 1. SETUP PATHS ---
base_dir = r"E:\deep\Dataset"
train_dir = os.path.join(base_dir, 'Train')
val_dir = os.path.join(base_dir, 'Validation')

# --- 2. DATA PREPARATION ---
# Use smaller target size for faster training on a laptop
IMG_SIZE = (128, 128) 
BATCH_SIZE = 32

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True, 
    zoom_range=0.2
)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')

val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')

# --- 3. BUILD MODEL ---
# Using MobileNetV2: Very fast and accurate for facial deepfakes
base_model = applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- 4. START TRAINING ---
print("\n🔥 Training is starting on your RTX 4050...")
model.fit(train_gen, validation_data=val_gen, epochs=10)

# --- 5. SAVE ---
model.save("deepfake_model_v1.h5")
print("\n✅ Training Complete! Model saved as deepfake_model_v1.h5")