import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the saved model
# Load the saved model with a corrected file path
model = load_model('efficientnetv2_gender_age_model.keras')


# Function to preprocess the image
def preprocess_image(img_path, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)  # Load and resize image
    img_array = img_to_array(img)  # Convert to numpy array
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1]
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

    # Check the output structure
    print("Predictions:", predictions)
    print("Shape of predictions:", predictions.shape)

    # Since predictions contain two values, assume they represent probabilities for gender
    predicted_gender_prob = predictions[0]  # Extract the first (and only) row of predictions

    # Predicted gender is the index of the higher probability: 0 = Female, 1 = Male
    predicted_gender = np.argmax(predicted_gender_prob)

    return None, predicted_gender  # Return None for age since it doesn't seem to be predicted




# Function to test the model on a folder of images, calculate accuracy/MAE, and plot confusion matrix
# Function to test the model on a folder of images, calculate accuracy, and plot confusion matrix
def evaluate_on_folder(folder_path):
    gender_correct = 0
    total_images = 0

    # Lists to store true and predicted labels for confusion matrix
    true_genders = []
    predicted_genders = []

    for img_file in os.listdir(folder_path):
        if img_file.lower().endswith(('png', 'jpg', 'jpeg')):  # Filter image files
            img_path = os.path.join(folder_path, img_file)

            # Get the actual age and gender from filename
            actual_age, actual_gender = extract_age_gender_from_filename(img_file)

            # Make predictions (note that predicted_age is ignored for now)
            _, predicted_gender = predict_single_image(img_path)

            # Compare gender (actual: 0 = Female, 1 = Male)
            true_genders.append(actual_gender)
            predicted_genders.append(predicted_gender)

            if predicted_gender == actual_gender:
                gender_correct += 1

            total_images += 1

            # Print individual image result (optional)
            gender_str = 'Male' if predicted_gender == 1 else 'Female'
            print(f"Image: {img_file}")
            print(f"Actual Gender: {'Male' if actual_gender == 1 else 'Female'}, Predicted Gender: {gender_str}")
            print("-" * 30)

    # Calculate overall accuracy for gender
    gender_accuracy = (gender_correct / total_images) * 100

    print(f"\nTotal Images: {total_images}")
    print(f"Gender Prediction Accuracy: {gender_accuracy:.2f}%")

    # Generate confusion matrix
    cm = confusion_matrix(true_genders, predicted_genders)

    # Print the confusion matrix to check values
    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Female", "Male"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Gender Classification')
    plt.show()



# Test on a folder of images
test_folder = '../../DataSet/test'  # Replace with your test folder path
evaluate_on_folder(test_folder)