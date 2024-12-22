import tensorflow as tf
import tensorflow_datasets as tfds
from datasets import load_dataset

def preprocess_data(image, label, img_size=(128, 128)):
    """
    Preprocess a single image and label.
    - Resizes the image
    - Normalizes pixel values to range [0, 1]
    """
    image = tf.image.resize(image, img_size) / 255.0
    return image, label

def load_data():
    """
    Load the TrashNet dataset and apply preprocessing.
    Splits data into training, validation, and test sets.
    """
    dataset = load_dataset("garythung/trashnet")
    train_dataset = dataset['train']

    # Create TensorFlow datasets
    tf_dataset = tf.data.Dataset.from_generator(
    lambda: (preprocess_data(item) for item in train_dataset),
    output_signature = (
        tf.TensorSpec(shape=(128,128,3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64)
        )
    )

    # Split into train, validation, and test sets
    train_size = int(0.7 * len(train_dataset))
    val_size = int(0.15 * len(train_dataset))
    test_size = len(train_dataset) - train_size - val_size

    train_data = tf_dataset.take(train_size)
    remaining = tf_dataset.skip(train_size)
    val_data = remaining.take(val_size)
    test_data = remaining.skip(val_size)

    # Batch and prefetch data
    batch_size = 32
    train_data = train_data.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_data = val_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_data = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE) 
    
    return train_data, val_data, test_data, len(train_dataset.features['label'].names)

if __name__ == "__main__":
    train_data, val_data, test_data, num_classes = load_data()
    print(f"Data loaded. Number of classes: {num_classes}")
