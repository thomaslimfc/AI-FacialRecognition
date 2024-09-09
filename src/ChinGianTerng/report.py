import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


# Define paths
test_dir = '../DataSet/test'  # Path to your test data

# Define constants
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

# Helper function to extract labels from filenames
def extract_labels_from_filename(filename):
    try:
        parts = filename.split('_')
        age = int(parts[0])  # Age
        gender = int(parts[1])  # Gender (0: male, 1: female)
        return age, gender
    except (IndexError, ValueError) as e:
        print(f"Error extracting labels from filename: {filename} - {e}")
        return None, None

# Data preparation functions
def load_and_preprocess_image(filepath, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

def create_dataset(filepaths, age_labels, gender_labels, batch_size=32, shuffle=True):
    def process_path(filepath, age, gender):
        img = load_and_preprocess_image(filepath)
        return img, (age, gender)  # Return both age and gender as labels

    filepaths_ds = tf.data.Dataset.from_tensor_slices((filepaths, age_labels, gender_labels))
    dataset = filepaths_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1024)  # Fixed buffer size
    dataset = dataset.batch(batch_size)

    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Prepare test dataset
test_filepaths = []
test_age_labels = []
test_gender_labels = []

for fname in os.listdir(test_dir):
    filepath = os.path.join(test_dir, fname)
    age, gender = extract_labels_from_filename(fname)

    if age is not None and gender is not None:
        test_filepaths.append(filepath)
        test_age_labels.append(age)
        test_gender_labels.append(gender)

test_dataset = create_dataset(test_filepaths, test_age_labels, test_gender_labels, batch_size=BATCH_SIZE, shuffle=False)

# Load the trained model
model_path = '../ChinGianTerng/vgg16_age_gender_model.h5'  # Replace with your model file
loaded_model = tf.keras.models.load_model(model_path)

# Make predictions on the test dataset
predictions = loaded_model.predict(test_dataset)

# Extract gender predictions (assuming predictions[1] is gender)
y_pred_gender = (predictions[1] > 0.5).astype(int)

# Extract true gender values from test dataset
y_true_gender = np.array([y[1].numpy() for _, y in test_dataset.unbatch()])

# Calculate precision, recall, and F1-score for gender
precision = precision_score(y_true_gender, y_pred_gender)
recall = recall_score(y_true_gender, y_pred_gender)
f1 = f1_score(y_true_gender, y_pred_gender)

# Print evaluation metrics
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_true_gender, y_pred_gender))