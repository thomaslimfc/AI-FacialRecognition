import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Set your image directory
train_data_dir = 'model/train_data'  # Replace with your training data path
test_data_dir = 'model/test_data'  # Replace with your test data path

# Image parameters
img_width, img_height = 224, 224  # You can adjust these dimensions based on your images
batch_size = 32
epochs = 10

# Data Augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Reserve 20% of training data for validation
)

# Preprocessing test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='sparse',  # Use 'sparse' for age prediction
    subset='training'  # Use subset for training split
)

# Load validation data
validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='sparse',  # Use 'sparse' for age prediction
    subset='validation'  # Use subset for validation split
)

# Load test data
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='sparse'
)

# Model definition
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='linear')  # Linear activation for age prediction
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Save the model in the recommended format
model.save('my_deepface_model.keras')  # Save in the new Keras format

# Evaluate the model on the test data
test_loss, test_mae = model.evaluate(test_generator)
print(f"Test MAE: {test_mae}")

# Predict on new images (replace with your images or test images)
# Here is an example of predicting using a single image
from tensorflow.keras.preprocessing import image

img = image.load_img('path/to/single_image.jpg', target_size=(img_width, img_height))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Reshape for prediction
img_array /= 255.0  # Rescale

# Predict the age
predicted_age = model.predict(img_array)
print(f"Predicted Age: {predicted_age[0][0]}")

