import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np

# Load the saved model
model = load_model('age_gender_model.keras')


# Function to preprocess the image
def preprocess_image(img_path, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)  # Load the image and resize it
    img_array = img_to_array(img)  # Convert the image to a numpy array
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image to [0, 1]
    return img_array


# Function to extract the actual age and gender from the filename
def extract_age_gender_from_filename(filename):
    # Assuming filename format is 'age_gender_image.jpg' where gender: 0 = Female, 1 = Male
    parts = filename.split('_')
    age = int(parts[0])
    gender = int(parts[1])
    return age, gender


# Function to make predictions on a single image and return results
def predict_single_image(img_path):
    img_array = preprocess_image(img_path)

    # Make predictions
    predictions = model.predict(img_array)

    # Extract predictions for age and gender
    predicted_age = predictions[0][0][0]  # Age output (remove list brackets)
    predicted_gender = predictions[1][0]  # Gender output (0 = Female, 1 = Male)

    return predicted_age, predicted_gender


# Function to test the model on a folder of images and calculate accuracy/MAE
def evaluate_on_folder(folder_path):
    gender_correct = 0
    total_images = 0
    age_mae_sum = 0

    for img_file in os.listdir(folder_path):
        if img_file.lower().endswith(('png', 'jpg', 'jpeg')):  # Filter image files
            img_path = os.path.join(folder_path, img_file)

            # Get the actual age and gender from filename
            actual_age, actual_gender = extract_age_gender_from_filename(img_file)

            # Make predictions
            predicted_age, predicted_gender = predict_single_image(img_path)

            # Compare gender (actual: 0 = Female, 1 = Male)
            predicted_gender_label = 1 if predicted_gender > 0.5 else 0
            if predicted_gender_label == actual_gender:
                gender_correct += 1

            # Calculate MAE for age
            age_mae_sum += abs(predicted_age - actual_age)

            total_images += 1

            # Print individual image result (optional)
            gender_str = 'Male' if predicted_gender_label == 1 else 'Female'
            print(f"Image: {img_file} | Actual Age: {actual_age}, Predicted Age: {predicted_age:.2f}")
            print(f"Actual Gender: {'Male' if actual_gender == 1 else 'Female'}, Predicted Gender: {gender_str}")
            print("-" * 30)

    # Calculate overall accuracy for gender and MAE for age
    gender_accuracy = (gender_correct / total_images) * 100
    age_mae = age_mae_sum / total_images

    print(f"\nTotal Images: {total_images}")
    print(f"Gender Prediction Accuracy: {gender_accuracy:.2f}%")
    print(f"Mean Absolute Error (MAE) for Age: {age_mae:.2f}")


# Test on a folder of images
test_folder = 'DataSet/test'  # Replace with your test folder path
evaluate_on_folder(test_folder)