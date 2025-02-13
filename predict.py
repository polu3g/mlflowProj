from __future__ import print_function
import os
import mlflow
import mlflow.keras
import numpy as np
from tensorflow.keras.datasets import mnist

# Function to load a model from MLflow and make predictions
def predict_from_model(registered_model_name, model_version=None):
    # Load the MNIST dataset for prediction
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test / 255.0  # Normalize data

    # Fetch the model from the MLflow Model Registry
    if model_version:
        print(f"Loading version {model_version} of the registered model '{registered_model_name}'.")
        model_uri = f"models:/{registered_model_name}/{model_version}"
    else:
        print(f"Loading the latest version of the registered model '{registered_model_name}'.")
        model_uri = f"models:/{registered_model_name}/latest"

    print("Model URI:", model_uri)

    # Load the model
    model = mlflow.keras.load_model(model_uri)
    print("Model successfully loaded from MLflow.")

    # Make predictions on the test data
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)

    # Print the first 10 predictions
    print("First 10 Predictions:")
    print("Predicted Classes:", predicted_classes[:10])
    print("Actual Classes:", y_test[:10])

    # Calculate the accuracy on the test set
    accuracy = np.mean(predicted_classes == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    return predictions, predicted_classes

if __name__ == "__main__":
    registered_model_name = "mnist_keras_model"  # Replace with your registered model name
    model_version = None  # Set a specific version if needed, e.g., "1"
    
    print("MLflow Version:", mlflow.version.VERSION)
    print("Tracking URI:", mlflow.tracking.get_tracking_uri())

    # Run the prediction function
    predictions, predicted_classes = predict_from_model(registered_model_name, model_version)
