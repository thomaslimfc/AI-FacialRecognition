import os
import tensorflow as tf
from dataset_preprocessing import prepare_data
from model_architecture import build_model
from tensorflow.keras.optimizers import Adam


def train_model():
    # Load preprocessed data
    dataset_path = "DataSet/train"
    print("Loading data...")
    X_train, X_val, y_age_train, y_age_val, y_gender_train, y_gender_val = prepare_data(dataset_path)
    print("Data loaded.")

    # Build the model (VGG16 or another base model)
    print("Building model...")
    model = build_model()
    print("Model built.")

    # Compile the model with different losses for age and gender
    print("Compiling model...")
    model.compile(optimizer=Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
                  loss={
                      'age_output': 'mean_squared_error',  # Regression for age
                      'gender_output': 'binary_crossentropy'  # Binary classification for gender
                  },
                  metrics={
                      'age_output': 'mae',  # Mean absolute error for age
                      'gender_output': 'accuracy'  # Accuracy for gender
                  })
    print("Model compiled.")

    # Train the model
    print("Starting model training...")
    history = model.fit(
        X_train,
        {'age_output': y_age_train, 'gender_output': y_gender_train},
        validation_data=(X_val, {'age_output': y_age_val, 'gender_output': y_gender_val}),
        epochs=20,
        batch_size=32
    )
    print("Training complete.")

    # Save the model
    model.save('vgg16_age_gender_model.h5')

    print("Model training completed and saved.")


if __name__ == "__main__":
    train_model()
