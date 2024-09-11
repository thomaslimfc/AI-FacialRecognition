import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set paths
dataset_path = 'DataSet/train'
labels = []

# Collect labels
for filename in os.listdir(dataset_path):
    if filename.endswith('.jpg'):
        parts = filename.split('_')
        if len(parts) >= 4:
            try:
                age = int(parts[0])
                gender = int(parts[1])
                labels.append((filename, age, gender))
            except ValueError:
                print(f"Skipping file {filename}: Invalid age or gender")
        else:
            print(f"Skipping file {filename}: Filename does not match expected format")

# Create DataFrame
df = pd.DataFrame(labels, columns=['filename', 'age', 'gender'])

# Split dataset
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the splits
train_df.to_csv('train_labels.csv', index=False)
val_df.to_csv('val_labels.csv', index=False)

# Define ImageDataGenerator
def get_datagen():
    return ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

train_datagen = get_datagen()
val_datagen = get_datagen()

# Define data generators
def get_data_generator(dataframe, subset):
    return train_datagen.flow_from_dataframe(
        dataframe=dataframe,
        directory=dataset_path,
        x_col='filename',
        y_col=['age', 'gender'],
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',  # Change this to 'raw'
        subset=subset
    )

train_generator = get_data_generator(train_df, 'training')
val_generator = get_data_generator(val_df, 'validation')
