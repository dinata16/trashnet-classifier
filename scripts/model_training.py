import tensorflow as tf
from tensorflow.keras import layers, models
from data_preparation import load_data

def build_model(input_shape=(128, 128, 3), num_classes=6):
    """
    Builds a CNN model using TensorFlow Keras.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation='softmax')
        ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model():
    """
    Load data, build the model, and train it.
    """
    train_data, val_data, _, num_classes = load_data()
    
    model = build_model(num_classes=num_classes)
    model.summary()
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=10
    )
    model.save('trashnet_model.h5')
    return history

if __name__ == "__main__":
    train_model()
