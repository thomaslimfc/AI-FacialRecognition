import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# 1. Extract Labels from Filename
def extract_labels(filename):
    parts = filename.split('_')
    age = int(parts[0])
    gender = 0 if parts[1].lower() == 'm' else 1
    return age, gender

# 2. Load and Modify the Pre-trained InceptionV3 Model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Add custom layers on top of InceptionV3
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# Define the output layers for age and gender
age_output = Dense(1, activation='linear', name='age')(x)  # Age is a regression task
gender_output = Dense(1, activation='sigmoid', name='gender')(x)  # Gender is a binary classification task

# Combine the model
model = Model(inputs=base_model.input, outputs=[age_output, gender_output])

# Freeze the layers of the base model to prevent them from being trained
for layer in base_model.layers:
    layer.trainable = False

# 3. Compile the Model
model.compile(
    optimizer='adam',
    loss={'age': 'mse', 'gender': 'binary_crossentropy'},  # Losses for age and gender
    metrics={'age': 'mae', 'gender': 'accuracy'}  # MAE for age, Accuracy for gender
)

# 4. Prepare a Custom Data Generator
class CustomDataGenerator(Sequence):
    def __init__(self, directory, batch_size=32, target_size=(299, 299), shuffle=True):
        self.directory = directory
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle

        # Verify and print the directory path
        absolute_path = os.path.abspath(directory)
        print(f"Using directory: {absolute_path}")

        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"Directory '{absolute_path}' does not exist.")

        # Load all image file names
        self.image_files = [f for f in os.listdir(absolute_path) if f.endswith('.jpg')]
        self.indexes = np.arange(len(self.image_files))
        self.on_epoch_end()

    def __len__(self):
        # Number of batches per epoch
        return int(np.floor(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_files = [self.image_files[k] for k in batch_indexes]

        images, age_labels, gender_labels = [], [], []
        for filename in batch_files:
            # Load and preprocess image
            image_path = os.path.join(self.directory, filename)
            img = load_img(image_path, target_size=self.target_size)
            img_array = img_to_array(img) / 255.0  # Normalize the image
            images.append(img_array)

            # Extract age and gender labels from the filename
            age, gender = extract_labels(filename)
            age_labels.append(age)
            gender_labels.append(gender)

        return np.array(images), [np.array(age_labels), np.array(gender_labels)]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Set the correct path to your dataset directory
train_generator = CustomDataGenerator(directory="../DataSet/train")

# 5. Train the Model
history = model.fit(
    train_generator,
    epochs=20,  # Set number of epochs
    steps_per_epoch=len(train_generator)  # Define the steps per epoch
)

# 6. Save the Model
model.save('inceptionv3_gender_age_model')  # Save in TensorFlow SavedModel format
