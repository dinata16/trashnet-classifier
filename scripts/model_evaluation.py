import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from data_preparation import load_data

def evaluate_model():
    """
    Loads the trained model and evaluates it on the test dataset.
    Generates a classification report and confusion matrix.
    """
    _, _, test_data, num_classes = load_data()
    model = tf.keras.models.load_model('trashnet_model.h5')
    
    y_true = []
    y_pred = []

    for images, labels in test_data:
        predictions = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
    
    print(f"Classification Report: \n{classification_report(y_true, y_pred)}")
    

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    evaluate_model()
