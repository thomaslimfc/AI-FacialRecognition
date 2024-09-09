import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(dataset_path):
    images = []
    age_labels = []
    gender_labels = []

    for img_name in os.listdir(dataset_path):
        # Filename format: [age]_[gender]_[race]_[date&time].jpg
        parts = img_name.split('_')
        age = int(parts[0])
        gender = int(parts[1])  # 0 for male, 1 for female

        # Load and preprocess the image
        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))  # VGG16 expects 224x224 input size
        images.append(img)
        age_labels.append(age)
        gender_labels.append(gender)

    images = np.array(images, dtype="float32") / 255.0  # Normalize the images
    age_labels = np.array(age_labels)
    gender_labels = np.array(gender_labels)

    return images, age_labels, gender_labels


def prepare_data(dataset_path, test_size=0.2):
    images, age_labels, gender_labels = load_data(dataset_path)

    # Split data into training and validation sets
    X_train, X_val, y_age_train, y_age_val = train_test_split(
        images, age_labels, test_size=test_size, random_state=42
    )

    _, _, y_gender_train, y_gender_val = train_test_split(
        images, gender_labels, test_size=test_size, random_state=42
    )

    return X_train, X_val, y_age_train, y_age_val, y_gender_train, y_gender_val
