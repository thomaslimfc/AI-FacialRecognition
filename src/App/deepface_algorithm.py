import cv2
from deepface import DeepFace
from ultralytics import YOLO

model = YOLO('yolov8n-face.pt')


def process_frame(frame):
    results = model(frame)
    faces_data = []

    for result in results:
        if result.boxes is not None:
            for box in result.boxes.data.tolist():
                x1, y1, x2, y2, score = map(float, box[:5])
                if score > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    face = frame[y1:y2, x1:x2]

                    # Convert face to the required format
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    result = DeepFace.analyze(face, actions=['age', 'gender'], enforce_detection=False)

                    # Print result for debugging
                    # print("Result type:", type(result))
                    # print("Result content:", result)

                    # Define gender mapping
                    gender_mapping = {'Man': 'Male', 'Woman': 'Female'}

                    if isinstance(result, list) and len(result) > 0:
                        age = str(result[0]['age'])
                        age_group = get_age_group(age)

                        gender_dict = result[0]['gender']
                        if isinstance(gender_dict, dict):
                            gender_dict = {gender_mapping.get(gender, gender): prob for gender, prob in
                                           gender_dict.items()}
                            gender, probability = max(gender_dict.items(), key=lambda item: item[1])
                            gender_formatted = f"{gender} {probability:.2f}%"
                        else:
                            gender_formatted = "unknown"
                    else:
                        age_group = "unknown"
                        gender_formatted = "unknown"

                    # Append data for this face to faces_data
                    faces_data.append((x1, y1, x2, y2, age_group, gender_formatted))

    # Return the collected face data
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
