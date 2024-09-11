import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Load the model directly from the .h5 file
try:
    model = tf.keras.models.load_model('efficientnetv2_gender_age_model.h5')
    print("Model loaded successfully from the .h5 file.")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)


# Function to preprocess the image
def preprocess_image(img_path, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image to [0, 1]
    return img_array


# Function to extract the actual age and gender from the filename
def extract_age_gender_from_filename(filename):
    parts = filename.split('_')
    age = int(parts[0])
    gender = int(parts[1])
    return age, gender


# Function to make predictions on a single image and return results
def predict_single_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)

    # Debugging: Print the predictions and its shape
    print("Predictions:", predictions)

    predicted_age = predictions[0][0]  # First output is for age
    predicted_gender = predictions[1][0]  # Second output is for gender
    return predicted_age, predicted_gender


# Function to test the model on a folder of images and calculate accuracy
def evaluate_on_folder(folder_path):
    gender_correct = 0
    total_images = 0

    for img_file in os.listdir(folder_path):
        if img_file.lower().endswith(('png', 'jpg', 'jpeg')):  # Filter image files
            img_path = os.path.join(folder_path, img_file)

            actual_age, actual_gender = extract_age_gender_from_filename(img_file)
            predicted_age, predicted_gender = predict_single_image(img_path)

            predicted_gender_label = 1 if predicted_gender > 0.5 else 0
            if predicted_gender_label == actual_gender:
                gender_correct += 1

            total_images += 1

            # Print individual image result
            gender_str = 'Male' if predicted_gender_label == 1 else 'Female'
            print(f"Image: {img_file} | Actual Gender: {'Male' if actual_gender == 1 else 'Female'}, "
                  f"Predicted Gender: {gender_str}, Predicted Age: {predicted_age:.2f}")
            print("-" * 30)

    # Calculate overall accuracy for gender
    gender_accuracy = (gender_correct / total_images) * 100
    print(f"\nTotal Images: {total_images}")
    print(f"Gender Prediction Accuracy: {gender_accuracy:.2f}%")


# Test on a folder of images
test_folder = '../DataSet/test'  # Replace with your test folder path
evaluate_on_folder(test_folder)
