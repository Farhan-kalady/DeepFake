import tensorflow as tf
model = tf.keras.models.load_model('deepfake_model_v1.h5')
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
        print(f"Last convolutional layer: {layer.name}")
        break
else:
    # Check if it's a functional model with nested layers
    for layer in reversed(model.layers):
        if hasattr(layer, 'layers'):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D) or 'conv' in sub_layer.name.lower():
                    print(f"Last convolutional layer in {layer.name}: {sub_layer.name}")
                    break
            else: continue
            break
