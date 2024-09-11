import tensorflow as tf

# Load the .h5 model
h5_model_path = 'age_gender_model.h5'
model = tf.keras.models.load_model(h5_model_path, compile=False)

# Save the model in the .keras format
keras_model_path = 'age_gender_model.keras'
model.save(keras_model_path)

print(f"Model converted and saved to {keras_model_path}")
