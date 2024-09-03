import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n-face.pt')  # Filepath to the model

# Initialize webcam
video_capture = cv2.VideoCapture(0)

print("Starting video stream. Press 'q' to quit.")

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = video_capture.read()

    # Use YOLOv8 model to detect faces
    results = model(frame)

    # Get the bounding boxes and confidence scores
    for result in results:
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, score = map(int, box[:5])
            if score > 0.5:  # Filter out weak detections
                # Draw rectangle around the face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Display the score
                cv2.putText(frame, f'{score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the resulting frame
    cv2.imshow('YOLOv8 Face Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
