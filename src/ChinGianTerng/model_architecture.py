from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model


def build_model():
    # Load VGG16 without the top layer (pre-trained on ImageNet)
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model layers to retain the pre-trained weights
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers for age and gender classification
    x = base_model.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)  # Add dropout for regularization

    # Age prediction branch
    age_output = Dense(1, activation='linear', name='age_output')(x)  # Regression for age

    # Gender prediction branch
    gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)  # Binary classification for gender

    # Create the model
    model = Model(inputs=base_model.input, outputs=[age_output, gender_output])

    return model
