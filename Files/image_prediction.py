import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
model = tf.keras.models.load_model('product_model.keras')

# Path of new image
img_path = r"sample.jpg"   # change this

IMG_SIZE = (224, 224)

class_names = np.load("class_names.npy", allow_pickle=True)
# ---------------------------
# Load and preprocess image
# ---------------------------
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)

# Normalize
img_array = img_array / 255.0

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# ---------------------------
# Predict
# ---------------------------
prediction = model.predict(img_array)

# Get class label
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction)

print("Predicted Category:", predicted_class)
print("Confidence:", confidence)