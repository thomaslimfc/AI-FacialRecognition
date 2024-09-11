import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import classification_report


# 1. Extract Labels from Filename
def extract_labels(filename):
    parts = filename.split('_')
    age = int(parts[0])
    gender = 0 if parts[1].lower() == 'm' else 1
    return age, gender


# 2. Load and Modify the Pre-trained EfficientNetV2B0 Model
base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top of EfficientNetV2B0
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
    def __init__(self, directory, batch_size=32, target_size=(224, 224), shuffle=True):
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

        # Print file names for debugging
        print(f"Loaded {len(self.image_files)} files from the directory.")

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


# 5. Class Weighting for Imbalance
# Calculate class weights based on the gender distribution
def calculate_class_weights(generator):
    total = len(generator.image_files)
    male_count = sum(1 for f in generator.image_files if '_m' in f)
    female_count = total - male_count

    print(f"Male count: {male_count}, Female count: {female_count}")

    if male_count == 0:
        male_weight = 1.0
    else:
        male_weight = total / (2 * male_count)

    if female_count == 0:
        female_weight = 1.0
    else:
        female_weight = total / (2 * female_count)

    class_weight = {
        0: male_weight,  # Male
        1: female_weight  # Female
    }

    return class_weight


# Set the correct path to your dataset directory
train_generator = CustomDataGenerator(directory="../DataSet/train")

# Calculate class weights
class_weights = calculate_class_weights(train_generator)

# 6. Train the Model with Class Weighting
history = model.fit(
    train_generator,
    epochs=20,  # Set number of epochs
    steps_per_epoch=len(train_generator),  # Define the steps per epoch
    class_weight=class_weights  # Apply class weights directly
)

# 7. Evaluate the Model
# Assuming you have a validation generator
val_generator = CustomDataGenerator(directory="../DataSet/val", shuffle=False)
predictions = model.predict(val_generator)

# Convert predictions to binary for classification report
gender_pred = np.round(predictions[1]).astype(int)

# Extract true labels from the generator
true_labels = []
for _, (age_labels, gender_labels) in val_generator:
    true_labels.extend(gender_labels)

# Generate classification report
print("Classification Report:")
print(classification_report(true_labels, gender_pred))

# 8. Save the Model in H5 Format
model.save('efficientnetv2_gender_age_model2.h5')
