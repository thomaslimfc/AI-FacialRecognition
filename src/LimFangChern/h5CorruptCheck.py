import tensorflow as tf
from tensorflow.keras.models import load_model

def check_h5_file(file_path):
    try:
        # Attempt to load the model
        model = load_model(file_path)
        print(f"Model loaded successfully from {file_path}")
        return True
    except Exception as e:
        print(f"Error loading model from {file_path}: {e}")
        return False

# Replace with your .h5 model file path
h5_file_path = 'deepfaceTraining.py'
if check_h5_file(h5_file_path):
    print("The .h5 file is not corrupted.")
else:
    print("The .h5 file is corrupted.")
