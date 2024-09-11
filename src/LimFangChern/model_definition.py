import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam


def create_model():
    base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    age_output = Dense(1, activation='linear', name='age')(x)
    gender_output = Dense(1, activation='sigmoid', name='gender')(x)

    model = Model(inputs=base_model.input, outputs=[age_output, gender_output])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss={'age': 'mean_squared_error', 'gender': 'binary_crossentropy'},
        metrics={'age': 'mae', 'gender': 'accuracy'}  # Ensure metrics are correctly set
    )

    return model
