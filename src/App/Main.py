import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Import the algorithm files
import resnet50_algorithm
import vgg16_algorithm
import deepface_algorithm

# Create the main window
root = tk.Tk()
root.title("Face Recognition and Detail Estimation System")
root.geometry("900x700")
root.configure(bg="#e0e0e0")

# Global variables for video capture and algorithm choice
video_capture = None
algorithm_choice = tk.StringVar(value="Select a model")
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

            face_count = 0
            result_str = ""
            faces_data = []

            # Call the appropriate algorithm based on user selection
            if algorithm_choice.get() == "ResNet50":
                faces_data = resnet50_algorithm.process_frame(frame)
            elif algorithm_choice.get() == "VGG16":
                faces_data = vgg16_algorithm.process_frame(frame)
            elif algorithm_choice.get() == "DeepFace":
                faces_data = deepface_algorithm.process_frame(frame)

            # Process the faces data returned by the algorithm
            for (x1, y1, x2, y2, age, gender) in faces_data:
                face_count += 1
                result_str += f'Face at ({x1},{y1}): Age: {age}, Gender: {gender}\n'
                # Draw rectangle and labels
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                draw_text_with_stroke(frame, f'Age: {age}, Gender: {gender}', (x1, y2 + 20), font_scale=0.7,
                                      color=(0, 255, 255), stroke_color=(0, 0, 0))

            # Update the results text
            results_text.set(f'{result_str}\nDetected Faces: {face_count}')

            # Convert the image from BGR to RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            label_video.imgtk = imgtk
            label_video.configure(image=imgtk)

        # Call process_frame again after x-time milliseconds
        root.after(50, process_frame)

# Start of UI app frame
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

combo_algorithms = ttk.Combobox(frame_controls, textvariable=algorithm_choice,
                                values=["ResNet50", "VGG16", "DeepFace"],
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
