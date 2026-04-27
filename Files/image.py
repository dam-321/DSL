import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------------------
# Dataset Path (single folder)
# ---------------------------
data_dir = r"dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ---------------------------
# Train Dataset (80%)
# ---------------------------
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# ---------------------------
# Validation Dataset (20%)
# ---------------------------
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
print("Classes:", class_names)

# ---------------------------
# Normalize
# ---------------------------
normalization_layer = layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# ---------------------------
# CNN Model
# ---------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

# ---------------------------
# Compile
# ---------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------------------
# Train
# ---------------------------
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# ---------------------------
# Evaluate
# ---------------------------
loss, acc = model.evaluate(val_dataset)
print("Validation Accuracy:", acc)
np.save("class_names.npy", class_names)
model.save("product_model.keras")
print("model saved successfully")