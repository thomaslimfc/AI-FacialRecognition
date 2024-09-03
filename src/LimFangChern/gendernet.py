import torch
import cv2
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

# Initialize models
yolo_model = YOLO('yolov8s.pt')  # Load pre-trained YOLOv8 model
age_model = torch.load('agenet.pth')  # Load pre-trained AgeNet model
gender_model = torch.load('gendernet.pth')  # Load pre-trained GenderNet model

# Define preprocessing transformation for age and gender models
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to predict age
def predict_age(face_image):
    face_tensor = transform(face_image).unsqueeze(0)
    with torch.no_grad():
        age_prediction = age_model(face_tensor)
    age = age_prediction.argmax().item()  # Modify this to handle your age model's output properly
    return age

# Function to predict gender
def predict_gender(face_image):
    face_tensor = transform(face_image).unsqueeze(0)
    with torch.no_grad():
        gender_prediction = gender_model(face_tensor)
    gender = "Male" if gender_prediction.argmax().item() == 1 else "Female"  # Assuming 1 is Male and 0 is Female
    return gender

# Capture from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection with YOLOv8
    results = yolo_model(frame)

    # Iterate over detected objects
    for result in results.xyxy[0]:  # Accessing bounding boxes from results
        x1, y1, x2, y2, confidence, label = result
        label = int(label.item())  # Converting tensor label to integer

        # Assuming label '0' corresponds to 'face'
        if label == 0:  # Adjust this based on your YOLO class mappings
            # Extract the face from the frame
            face = frame[int(y1):int(y2), int(x1):int(x2)]
            face_image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

            # Predict age and gender for the detected face
            age = predict_age(face_image)
            gender = predict_gender(face_image)

            # Draw bounding box and display age and gender
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {age}", (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Gender: {gender}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Facial Recognition System', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
