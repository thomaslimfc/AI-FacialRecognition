import tensorflow as tf

# Load the .h5 model
model = tf.keras.models.load_model('my_deepface_model.h5')

# Save it as a .keras model
model.save('my_deepface_model.keras')

print("Model has been successfully converted to .keras format.")
