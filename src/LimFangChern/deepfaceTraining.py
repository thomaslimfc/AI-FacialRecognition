import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16  # Example using VGG16

# Load CSV labels
df = pd.read_csv('deepface.csv')

# Verify column names in the CSV file
print("Columns in CSV file:", df.columns)

# Ensure your CSV file has 'filename' and 'gender' columns
if 'filename' not in df.columns or 'gender' not in df.columns:
    raise ValueError("CSV file must contain 'filename' and 'gender' columns")

# Map gender to categorical labels if it's not already in categorical format
df['gender'] = df['gender'].astype(str)  # Ensure gender is a string for categorical conversion

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Debugging: Print a sample of filenames
print("Sample training filenames:", train_df['filename'].head())
print("Sample testing filenames:", test_df['filename'].head())

# Define image data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Define function to validate image paths
def validate_image_paths(df, directory):
    missing_files = []
    for filename in df['filename']:
        file_path = os.path.join(directory, filename)
        if not os.path.isfile(file_path):
            missing_files.append(file_path)
    if missing_files:
        print("Missing files:")
        for file in missing_files:
            print(file)
    else:
        print("All files are present.")

# Validate image paths for training and testing datasets
validate_image_paths(train_df, '../DataSet/train')
validate_image_paths(test_df, '../DataSet/test')

# Set up training data generator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=os.path.abspath('../DataSet/train'),
    x_col='filename',
    y_col='gender',
    target_size=(224, 224),  # Adjust size if needed
    batch_size=32,
    class_mode='categorical'
)

# Set up testing data generator
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=os.path.abspath('../DataSet/test'),
    x_col='filename',
    y_col='gender',
    target_size=(224, 224),  # Adjust size if needed
    batch_size=32,
    class_mode='categorical'
)

# Debugging: Check number of samples in generators
print("Training samples:", train_generator.samples)
print("Testing samples:", test_generator.samples)

# Define the model using VGG16 as a base
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator
)

# Save the trained model
model.save('my_deepface_model.h5')
