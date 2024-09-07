import tensorflow as tf


def load_dataset(directory, batch_size=32, image_size=(224, 224)):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='int',  # Use 'int' if you want integer labels
        image_size=image_size,
        batch_size=batch_size
    )
    dataset = dataset.map(lambda x, y: (x / 255.0, y))  # Normalize images
    return dataset


train_dataset = load_dataset('dataset/train', batch_size=32, image_size=(224, 224))
test_dataset = load_dataset('dataset/test', batch_size=32, image_size=(224, 224))
