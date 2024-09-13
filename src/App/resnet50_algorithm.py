import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from ultralytics import YOLO

# Load the saved model
model = load_model('../GohWeiZhun/ResNet50.keras')

# Load YOLO model for face detection
yolo_model = YOLO('yolov8n-face.pt')

# Function to preprocess the face image for ResNet50 model
def preprocess_face(face, target_size=(224, 224)):
    face = cv2.resize(face, target_size)  # Resize face to target size
    face_array = img_to_array(face)  # Convert face to a numpy array
    face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension
    face_array /= 255.0  # Normalize the face to [0, 1]
    return face_array

# Function to predict age and gender using the ResNet50 model
def predict_age_gender(face):
    face_array = preprocess_face(face)
    predictions = model.predict(face_array)

    predicted_age = predictions[0][0][0]  # Age output
    predicted_gender = predictions[1][0]  # Gender output (0 = Female, 1 = Male)

    # Convert the numerical gender value to a readable label
    if predicted_gender > 0.09:
        predicted_gender_label = 'Male'
    else:
        predicted_gender_label = 'Female'

    # Get the age group
    predicted_age_group = get_age_group(predicted_age)

    return predicted_age_group, predicted_gender_label  # Return age group and gender label

def get_age_group(age):
    """Convert age to an age group."""
    age = float(age)
    if 0 <= age < 5:
        return "0-4"
    elif 5 <= age < 10:
        return "5-9"
    elif 10 <= age < 15:
        return "10-14"
    elif 15 <= age < 20:
        return "15-19"
    elif 20 <= age < 25:
        return "20-24"
    elif 25 <= age < 30:
        return "25-29"
    elif 30 <= age < 35:
        return "30-34"
    elif 35 <= age < 40:
        return "35-39"
    elif 40 <= age < 45:
        return "40-44"
    elif 45 <= age < 50:
        return "45-49"
    elif 50 <= age < 55:
        return "50-54"
    elif 55 <= age < 60:
        return "55-59"
    elif 60 <= age < 65:
        return "60-64"
    elif 65 <= age < 70:
        return "65-69"
    elif 70 <= age < 75:
        return "70-74"
    elif 75 <= age < 80:
        return "75-79"
    elif 80 <= age < 85:
        return "80-84"
    elif 85 <= age < 90:
        return "85-89"
    else:
        return "90+"


# Function to detect faces using YOLO and predict age/gender for each face
def process_frame(frame):
    results = yolo_model(frame)
    faces_data = []

    # Loop over YOLO-detected faces
    for result in results:
        for box in result.boxes:
            # Extract the face bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Extract the face region from the frame
            face = frame[y1:y2, x1:x2]

            # Predict age and gender for the face
            predicted_age_group, predicted_gender = predict_age_gender(face)

            # Store the result
            faces_data.append((x1, y1, x2, y2, predicted_age_group, predicted_gender))

            # Print the detected face information
            print(f"Face detected at ({x1}, {y1}, {x2}, {y2})")
            print(f"Predicted Age Group: {predicted_age_group}")
            print(f"Predicted Gender: {predicted_gender}")  # Show gender as Male/Female

    return faces_data