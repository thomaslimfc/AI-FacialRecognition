import os
import re
import shutil
import pandas as pd

from sklearn.model_selection import train_test_split

# Directories
source_dir = 'resized_images'
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# Create necessary directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(os.path.join(train_dir, 'age'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'gender'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'age'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'gender'), exist_ok=True)

# Get list of filenames
all_files = os.listdir(source_dir)

# Extract labels from filenames
data = []
pattern = re.compile(r'(\d+)_(\d)_(\d)_(\d+)\.jpg')
for filename in all_files:
    match = pattern.match(filename)
    if match:
        age, gender, race, _ = match.groups()
        age = int(age)
        gender = 'male' if int(gender) == 0 else 'female'
        data.append([filename, age, gender])

# Convert to DataFrame
df = pd.DataFrame(data, columns=['filename', 'age', 'gender'])

# Split data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


# Function to move files
def move_files(df, base_dir):
    for _, row in df.iterrows():
        filename = row['filename']
        age_dir = os.path.join(base_dir, 'age', str(row['age']))
        gender_dir = os.path.join(base_dir, 'gender', row['gender'])

        # Create directories if they don't exist
        os.makedirs(age_dir, exist_ok=True)
        os.makedirs(gender_dir, exist_ok=True)

        # Move files
        src_path = os.path.join(source_dir, filename)
        dst_path_age = os.path.join(age_dir, filename)
        dst_path_gender = os.path.join(gender_dir, filename)

        shutil.copy(src_path, dst_path_age)
        shutil.copy(src_path, dst_path_gender)


# Move training and testing files
move_files(train_df, train_dir)
move_files(test_df, test_dir)

print("Files organized successfully!")
