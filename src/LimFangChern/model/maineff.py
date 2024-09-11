import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve

# 1. Extract Labels from Filename
def extract_labels(filename):
    parts = filename.split('_')
    age = int(parts[0])
    gender = 0 if parts[1].lower() == 'm' else 1
    return age, gender

# 2. Define the Model
def create_model():
    base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    age_output = Dense(1, activation='linear', name='age')(x)
    gender_output = Dense(1, activation='sigmoid', name='gender')(x)

    model = Model(inputs=base_model.input, outputs=[age_output, gender_output])

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer='adam',
        loss={'age': 'mse', 'gender': 'binary_crossentropy'},
        metrics={'age': 'mae', 'gender': 'accuracy'}
    )

    return model

# 3. Define the Custom Data Generator
class CustomDataGenerator(Sequence):
    def __init__(self, directory, batch_size=32, target_size=(224, 224), shuffle=True):
        self.directory = directory
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle

        absolute_path = os.path.abspath(directory)
        print(f"Using directory: {absolute_path}")

        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"Directory '{absolute_path}' does not exist.")

        self.image_files = [f for f in os.listdir(absolute_path) if f.endswith('.jpg')]
        self.indexes = np.arange(len(self.image_files))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_files = [self.image_files[k] for k in batch_indexes]

        images, age_labels, gender_labels = [], [], []
        for filename in batch_files:
            image_path = os.path.join(self.directory, filename)
            img = load_img(image_path, target_size=self.target_size)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)

            age, gender = extract_labels(filename)
            age_labels.append(age)
            gender_labels.append(gender)

        return np.array(images), [np.array(age_labels), np.array(gender_labels)]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# 4. Main Training Function
def main():
    model = create_model()

    train_generator = CustomDataGenerator(directory="../DataSet/train")

    # Train the model
    history = model.fit(
        train_generator,
        epochs=20,
        steps_per_epoch=len(train_generator)
    )

    # Save the model
    model.save('efficientnetv2_gender_age_model.h5')

    # Optionally, evaluate precision and recall
    # Here, you need to provide x_test and y_test
    # y_proba = model.predict(x_test)
    # y_proba = y_proba[1]
    # precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    # best_threshold = thresholds[np.argmax(precision)]
    # y_pred = (y_proba >= best_threshold).astype(int)
    # Print or log precision and recall

if __name__ == "__main__":
    main()
