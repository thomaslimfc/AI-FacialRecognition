import tensorflow as tf
from model_definition import create_model
from dataset_preparation import train_generator, val_generator


def train_model():
    model = create_model()

    # Set up training
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,  # Adjust as needed
        verbose=2  # Use verbose=2 for detailed output
    )

    # Save the model
    model.save('age_gender_model.h5')

    # Print training history
    print("Training history:")
    print(f"Training Loss: {history.history['loss']}")
    print(f"Training Age MAE: {history.history['age_mae']}")
    print(f"Training Gender Accuracy: {history.history['gender_accuracy']}")
    print(f"Validation Loss: {history.history['val_loss']}")
    print(f"Validation Age MAE: {history.history['val_age_mae']}")
    print(f"Validation Gender Accuracy: {history.history['val_gender_accuracy']}")


if __name__ == '__main__':
    train_model()