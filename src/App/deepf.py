import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np  # Ensure numpy is imported
from keras.src.saving import load_model
from ultralytics import YOLO

# Initialize YOLO model for face detection
yolo_model = YOLO('yolov8n-face.pt')

# Load the fine-tuned VGG16 model
efficientNetV2_model = load_model('../LimFangChern/age_gender_model.keras')


# Preprocess the detected face for VGG16 input
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img


# Create the main window
root = tk.Tk()
root.title("Face Detection and Age/Gender Prediction")
root.geometry("900x700")
root.configure(bg="#e0e0e0")

# Global variables for video capture and algorithm choice
video_capture = None
algorithm_choice = tk.StringVar(value="DeepFace")
results_text = tk.StringVar(value="Results will be displayed here.")


def start_camera():
    global video_capture
    video_capture = cv2.VideoCapture(0)
    process_frame()


def stop_camera():
    global video_capture
    if video_capture is not None:
        video_capture.release()
        video_capture = None
        label_video.config(image='')
        results_text.set("Results will be displayed here.")


def process_frame():
    global video_capture
    ret, frame = video_capture.read()
    if not ret:
        return

    faces_data = detect_and_predict(frame)

    # Display the results
    display_results(faces_data, frame)

    # Update the video feed every 10ms
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    label_video.imgtk = imgtk
    label_video.config(image=imgtk)
    label_video.after(10, process_frame)


def detect_and_predict(frame):
    results = yolo_model(frame)
    faces_data = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_crop = frame[y1:y2, x1:x2]
            face_input = preprocess_face(face_crop)
            age_gender_prediction = efficientNetV2_model.predict(face_input)

            predicted_gender = np.argmax(age_gender_prediction[1])
            predicted_age = age_gender_prediction[0][0]

            age_group = get_age_group(predicted_age)
            gender = 'Male' if predicted_gender == 0 else 'Female'

            faces_data.append((x1, y1, x2, y2, age_group, gender))

    return faces_data


def display_results(faces_data, frame):
    results = []
    for (x1, y1, x2, y2, age_group, gender) in faces_data:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f'{gender}, {age_group}'
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        results.append(text)

    results_text.set('\n'.join(results))

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

# Modern style
style = ttk.Style()
style.configure('TFrame', background='#f5f5f5')
style.configure('TLabelFrame', background='#f5f5f5', font=('Helvetica', 12, 'bold'))
style.configure('TButton', background='#4CAF50', foreground='white', font=('Helvetica', 10, 'bold'))
style.map('TButton', background=[('active', '#45a049')])

# Create GUI elements
frame_controls = ttk.Frame(root, padding="10")
frame_controls.pack(pady=10, fill="x")

btn_start = ttk.Button(frame_controls, text="Start Camera", command=start_camera)
btn_start.grid(row=0, column=0, padx=10, pady=10)

btn_stop = ttk.Button(frame_controls, text="Stop Camera", command=stop_camera)
btn_stop.grid(row=0, column=1, padx=10, pady=10)

label_algorithms = ttk.Label(frame_controls, text="Select Algorithm:")
label_algorithms.grid(row=1, column=0, padx=10, pady=10)

combo_algorithms = ttk.Combobox(frame_controls, textvariable=algorithm_choice, values=["KNN", "SVM", "CNN"],
                                state="readonly")
combo_algorithms.grid(row=1, column=1, padx=10, pady=10)

frame_video = ttk.LabelFrame(root, text="Live Video Feed", padding="10")
frame_video.pack(padx=10, pady=10, fill="both", expand=True)

label_video = ttk.Label(frame_video)
label_video.pack()

frame_results = ttk.LabelFrame(root, text="Detection Results", padding="10")
frame_results.pack(padx=10, pady=10, fill="both", expand=True)

label_results = ttk.Label(frame_results, textvariable=results_text, justify="left", anchor="nw")
label_results.pack(padx=5, pady=5, fill="both", expand=True)

root.mainloop()
