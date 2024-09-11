import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten

# Recreate the model architecture
base_model = EfficientNetV2B0(weights=None, include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)

age_output = Dense(1, activation='linear', name='age')(x)
gender_output = Dense(1, activation='sigmoid', name='gender')(x)

model = Model(inputs=base_model.input, outputs=[age_output, gender_output])

# Compile the model (required before loading weights)
model.compile(
    optimizer='adam',
    loss={'age': 'mse', 'gender': 'binary_crossentropy'},
    metrics={'age': 'mae', 'gender': 'accuracy'}
)

# Load the weights from the .h5 file
model.load_weights('../age_gender_model.h5')

# Save the model as .keras format
model.save('../age_gender_model.keras')

print("Model has been successfully converted to .keras format.")
