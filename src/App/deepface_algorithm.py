import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from keras.src.saving import load_model
from ultralytics import YOLO


# Initialize YOLO model for face detection
yolo_model = YOLO('yolov8n-face.pt')

# Load the fine-tuned VGG16 model
efficientNetV2_model = load_model('../LimFangChern/age_gender_model.keras')


# Preprocess the detected face for VGG16 input
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (224, 224))  # Resize to the input size of VGG16
    face_img = face_img.astype('float32') / 255.0  # Normalize the image
    face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension
    return face_img


def process_frame(frame):
    # Detect faces using YOLOv8
    results = yolo_model(frame)
    faces_data = []

    # Iterate over detected faces
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop the face from the frame
            face_crop = frame[y1:y2, x1:x2]

            # Preprocess the face for VGG16 model input
            face_input = preprocess_face(face_crop)

            # Perform age and gender estimation using VGG16
            age_gender_prediction = efficientNetV2_model.predict(face_input)

            # Assuming the model has two heads: one for gender, one for age
            predicted_gender = np.argmax(age_gender_prediction[0])  # Gender: 0 = Male, 1 = Female
            predicted_age = age_gender_prediction[0][0][0]  # Age estimate (regression)
            # print predicted age and gender
            print(predicted_age)
            print(predicted_gender)
            # Convert age to an age group
            age_group = get_age_group(predicted_age)
            gender = 'Male' if predicted_gender == 0 else 'Female'

            # Append the detected data
            faces_data.append((x1, y1, x2, y2, age_group, gender))

    return faces_data


def get_age_group(age):
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
