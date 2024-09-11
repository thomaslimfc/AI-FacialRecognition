import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
from deepface import DeepFace

# Initialize YOLO model for face detection
model = YOLO('model/yolov8n-face.pt')  # Ensure this is the correct path to your YOLO model

# Create the main window
root = tk.Tk()
root.title("YOLOv8 for Facial Recognition with Age and Gender Prediction")
root.geometry("900x700")
root.configure(bg="#e0e0e0")  # Light grey background for modern look

# Global variables for video capture and algorithm choice
video_capture = None
algorithm_choice = tk.StringVar(value="KNN")  # Default to KNN
results_text = tk.StringVar(value="Results will be displayed here.")


def list_video_devices():
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Device index {index} is available")
            cap.release()
        else:
            break
        index += 1

list_video_devices()


def start_camera():
    global video_capture
    video_capture = cv2.VideoCapture(1)
    # index 0: laptop original webcam
    # video_capture = cv2.VideoCapture(0)
    process_frame()


def stop_camera():
    global video_capture
    if video_capture is not None:
        video_capture.release()
        video_capture = None
        # Clear the video display and results
        label_video.config(image='')
        results_text.set("Results will be displayed here.")


def draw_text_with_stroke(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.9, color=(0, 255, 255),
                          stroke_color=(0, 0, 0), stroke_thickness=2):
    # Draw the stroke effect by drawing text multiple times with a slight offset
    x, y = position
    for dx in range(-stroke_thickness, stroke_thickness + 1):
        for dy in range(-stroke_thickness, stroke_thickness + 1):
            if dx != 0 or dy != 0:
                cv2.putText(image, text, (x + dx, y + dy), font, font_scale, stroke_color, 2, cv2.LINE_AA)

    # Draw the actual text on top
    cv2.putText(image, text, position, font, font_scale, color, 2, cv2.LINE_AA)

def process_frame():
    global video_capture

    if video_capture is not None and video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Mirror the frame

            # Use YOLO for face detection
            results = model(frame)

            faces = []
            face_count = 0
            result_str = ""

            # Process each detected face
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes.data.tolist():
                        x1, y1, x2, y2, score = map(float, box[:5])
                        if score > 0.5:  # Confidence threshold
                            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            face = frame[y1:y2, x1:x2]
                            faces.append((face, (x1, y1, x2, y2)))
                            face_count += 1
                            # Remove or comment out this line to not display the confidence score
                            # draw_text_with_stroke(frame, f'{score:.2f}', (x1, y1 - 10), font_scale=0.9, color=(0, 255, 255), stroke_color=(0, 0, 0))

            # Apply the selected algorithm for each detected face
            for face, (x1, y1, x2, y2) in faces:
                face_resized = cv2.resize(face, (128, 128))

                if algorithm_choice.get() == "KNN":
                    age, gender = predict_age_gender_knn(face_resized)
                elif algorithm_choice.get() == "SVM":
                    age, gender = predict_age_gender_svm(face_resized)
                elif algorithm_choice.get() == "DeepFace":
                    age, gender = deepfacePrediction(face)
                else:
                    age, gender = "N/A", "N/A"

                result_str += f'Face at ({x1},{y1}): Age: {age}, Gender: {gender}\n'
                draw_text_with_stroke(frame, f'Age: {age}, Gender: {gender}', (x1, y2 + 20), font_scale=0.7, color=(0, 255, 255), stroke_color=(0, 0, 0))

            # Update the results text
            results_text.set(f'{result_str}\nDetected Faces: {face_count}')

            # Convert the image from BGR to RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            label_video.imgtk = imgtk
            label_video.configure(image=imgtk)

        # Call process_frame again after 10 ms
        root.after(10, process_frame)


def predict_age_gender_knn(face):
    return "25-30", "Male"  # Placeholder return values


def predict_age_gender_svm(face):
    return "20-25", "Female"  # Placeholder return values


from deepface import DeepFace
import cv2


# LimFangChern
from deepface import DeepFace
import cv2

def deepfacePrediction(face):
    # Convert face to the required format
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    result = DeepFace.analyze(face, actions=['age', 'gender'], enforce_detection=False)

    # Print result for debugging
    print("Result type:", type(result))
    print("Result content:", result)

    # Define gender mapping
    gender_mapping = {
        'Man': 'Male',
        'Woman': 'Female'
    }

    # Assuming result is a list of dictionaries
    if isinstance(result, list) and len(result) > 0:
        # Extract and format age group
        age = str(result[0]['age'])
        age_group = get_age_group(age)

        # Extract and format gender
        gender_dict = result[0]['gender']
        if isinstance(gender_dict, dict):
            # Map gender terms and get the highest probability
            gender_dict = {gender_mapping.get(gender, gender): prob for gender, prob in gender_dict.items()}
            gender, probability = max(gender_dict.items(), key=lambda item: item[1])
            gender_formatted = f"{gender} {probability:.2f}%"
        else:
            gender_formatted = "unknown"

    else:
        age_group = "unknown"
        gender_formatted = "unknown"

    return age_group, gender_formatted


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
style.configure('TButton', background='#4CAF50', foreground='black', font=('Helvetica', 10, 'bold'))
style.map('TButton', background=[('active', '#45a049')])

# Create GUI elements
frame_controls = ttk.Frame(root, padding="10")
frame_controls.pack(pady=10, fill="x")

# Design of Start Camera & Stop Camera button
style.configure(
    'OnclickButton.TButton',
    background='#4CAF50',
    foreground='black',
    font=('Helvetica', 10, 'bold'),
    padding=(10, 5),
    relief='solid',
    bordercolor='black',
    borderwidth=2
)
style.map(
    'OnclickButton.TButton',
    background=[('active', '#45a049')],
    cursor=[('hover', 'hand2')]
)

btn_start = ttk.Button(frame_controls, text="Start Camera", command=start_camera, style='OnclickButton.TButton')
btn_start.grid(row=0, column=0, padx=10, pady=10)

btn_stop = ttk.Button(frame_controls, text="Stop Camera", command=stop_camera, style='OnclickButton.TButton')
btn_stop.grid(row=0, column=1, padx=10, pady=10)

label_algorithms = ttk.Label(frame_controls, text="Select Algorithm:")
label_algorithms.grid(row=1, column=0, padx=10, pady=10)

# Combobox Design
style.configure(
    'TCombobox',
    foreground='black',
    background='white',
    fieldbackground='white',
    font=('Helvetica', 12),
)
style.map(
    'TCombobox',
    background=[('active', '#e0e0e0')],
    fieldbackground=[('readonly', 'white')],
    foreground=[('readonly', 'black')],
    bordercolor=[('focus', 'black')],
    arrowcolor=[('hover', 'black')]
)

combo_algorithms = ttk.Combobox(frame_controls, textvariable=algorithm_choice, values=["KNN", "SVM", "DeepFace"],
                                state="readonly", style='TCombobox')
combo_algorithms.grid(row=1, column=1, padx=20, pady=20)

# Video display frame
frame_video = ttk.LabelFrame(root, text="Live Video Feed", padding="10")
frame_video.pack(padx=10, pady=10, fill="both", expand=True)

label_video = ttk.Label(frame_video)
label_video.pack()

# Results display frame
frame_results = ttk.LabelFrame(root, text="Detection Results", padding="10")
frame_results.pack(padx=10, pady=10, fill="both", expand=True)

label_results = ttk.Label(frame_results, textvariable=results_text, justify="left", anchor="nw")
label_results.pack(padx=5, pady=5, fill="both", expand=True)

# Start the GUI main loop
root.mainloop()
