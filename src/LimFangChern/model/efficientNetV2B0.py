import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import img_to_array, load_img


# 1. Extract Labels from Filename
def extract_labels(filename):
    parts = filename.split('_')
    age = int(parts[0])  # The first part is the age
    gender = int(parts[1])  # The second part is the gender (1 = male, 0 = female)
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
            age_labels.append(float(age))  # Ensure age labels are float
            gender_labels.append(float(gender))  # Ensure gender labels are float

        # Convert lists to numpy arrays
        images = np.array(images)
        age_labels = np.array(age_labels)
        gender_labels = np.array(gender_labels)

        return images, {'age': age_labels, 'gender': gender_labels}

    def on_epoch_end(self):
        # Shuffle indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)


# 5. Create a TensorFlow Dataset from the Generator
def generator_function(directory, batch_size, target_size):
    def generator():
        custom_gen = CustomDataGenerator(directory, batch_size=batch_size, target_size=target_size)
        for images, labels in custom_gen:
            yield images, labels

    return generator


# Set the correct path to your dataset directory
train_dataset = tf.data.Dataset.from_generator(
    generator_function("../DataSet/train", batch_size=32, target_size=(224, 224)),
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),  # For images
        {
            'age': tf.TensorSpec(shape=(None,), dtype=tf.float32),  # For age labels
            'gender': tf.TensorSpec(shape=(None,), dtype=tf.float32)  # For gender labels
        }
    )
)

# Prefetch and batch the dataset
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# 6. Train the Model
history = model.fit(
    train_dataset,
    epochs=20,
    steps_per_epoch=len(CustomDataGenerator(directory="../DataSet/train"))
)

# 7. Save the Model
model.save('finalmodel.keras')  # Save the model
