import os
import pandas as pd

# Function to extract labels from filenames
def extract_labels_from_filenames(image_folder):
    data = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg'):
            parts = filename.split('_')
            if len(parts) == 4:  # If filename includes age, gender, race, and datetime
                age = parts[0]
                gender = 'Male' if parts[1] == '0' else 'Female'
                race = ['White', 'Black', 'Asian', 'Indian', 'Others'][int(parts[2])]
                data.append([filename, age, gender, race])
            elif len(parts) == 3:  # If filename has age, gender, and datetime (no race info)
                age = parts[0]
                gender = 'Male' if parts[1] == '0' else 'Female'
                race = 'Unknown'  # Default race when not available
                data.append([filename, age, gender, race])
            else:
                print(f"Skipping file: {filename}, not enough parts")
    return pd.DataFrame(data, columns=['filename', 'age', 'gender', 'race'])

# Example usage
df = extract_labels_from_filenames('../../DataSet/train')
df.to_csv('deepfaceTrain.csv', index=False)
df = extract_labels_from_filenames('../../DataSet/test')
df.to_csv('deepfaceTest.csv', index=False)
