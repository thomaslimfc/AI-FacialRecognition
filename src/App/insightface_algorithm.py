import cv2
from insightface.app import FaceAnalysis
from ultralytics import YOLO

# Initialize InsightFace and YOLO
app = FaceAnalysis(allowed_modules=['detection', 'genderage'])
app.prepare(ctx_id=0, det_thresh=0.5)
yolo_model = YOLO('yolov8n-face.pt')


def process_frame(frame):
    # Detect faces using YOLO
    results = yolo_model(frame)
    faces_data = []

    # Iterate over detected faces
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = app.get(frame_rgb)

            if len(faces) > 0:
                face = faces[0]
                age = face.age
                gender = 'Male' if face.gender == 1 else 'Female'
                # Convert age to age group
                age_group = get_age_group(age)

                # Append the detected data
                faces_data.append((x1, y1, x2, y2, age_group, gender))

    return faces_data


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
