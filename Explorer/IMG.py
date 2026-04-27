# ==========================================
# PLANT DISEASE IMAGE CLASSIFICATION (CNN)
# ==========================================

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ==========================================
# 1. LOAD DATA (FOLDER STRUCTURE)
# ==========================================

data_dir = "D:\Plantvillage_Disease\PlantVillage"

img_size = (128, 128)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Classes:", class_names)

# ==========================================
# 2. NORMALIZATION
# ==========================================
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# ==========================================
# 3. CNN MODEL
# ==========================================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================
# 4. TRAIN MODEL
# ==========================================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# ==========================================
# 5. EVALUATION
# ==========================================
loss, acc = model.evaluate(val_ds)
print("\nValidation Accuracy:", acc)
np.save("class_names.npy", class_names)
model.save("product_model.keras")
print("model saved successfully")

# ==========================================
# 6. PREDICTIONS
# ==========================================
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# ==========================================
# 7. METRICS
# ==========================================
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

# ==========================================
# 8. CONFUSION MATRIX PLOT
# ==========================================
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ==========================================
# 9. TRAINING GRAPH
# ==========================================
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Accuracy")
plt.show()

# ==========================================
# 10. TEST SINGLE IMAGE
# ==========================================
img_path = "D:\\Plantvillage_Disease\\PlantVillage\\Tomato__Tomato_YellowLeaf__Curl_Virus\\0a7f1993-85ae-4ba4-865b-df96bb0b52b4___YLCV_NREC 2416.JPG"   # give your image

img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128,128))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

pred = model.predict(img_array)
pred_class = class_names[np.argmax(pred)]

print("Predicted Class:", pred_class)