from __future__ import print_function
import os
import sys
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam


# Function to train a Keras model, log metrics, and register the model
def run(alpha, run_origin, log_artifact):
    mlflow.set_experiment(experiment_name)
    
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize data

    with mlflow.start_run(run_name=run_origin) as run:
        print("Run ID:", run.info.run_uuid)
        print("Artifact URI:", mlflow.get_artifact_uri())
        print("Alpha (Learning Rate):", alpha)
        print("Run Origin:", run_origin)
        
        # MLflow Parameters
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("epochs", 5)
        mlflow.log_param("batch_size", 64)
        
        # Build a simple neural network model
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        
        optimizer = Adam(learning_rate=float(alpha))
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        # Train the model
        history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
        
        # Log metrics to MLflow
        for epoch in range(5):
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
        
        # Log the model as an artifact
        if log_artifact:
            model.save("keras_model.h5")
            mlflow.log_artifact("keras_model.h5")
            print("Model saved and logged as an artifact.")
        
        # Log the model using MLflow's Keras logging
        mlflow.keras.log_model(model, "keras_model")
        print("Model logged to MLflow.")
        
        # Register the model in the MLflow Model Registry
        model_name = "mnist_keras_model"
        registered_model = mlflow.register_model(
            "runs:/{}/keras_model".format(run.info.run_id), model_name
        )
        
        # Add version details
        print("Registered Model Name:", registered_model.name)
        print("Registered Model Version:", registered_model.version)


if __name__ == "__main__":
    # Default values for arguments
    alpha = "0.001"  # Default learning rate
    run_origin = "tensorflow_run"
    log_artifact = True
    
    print("TensorFlow Version:", tf.__version__)
    print("MLflow Version:", mlflow.version.VERSION)
    print("Tracking URI:", mlflow.tracking.get_tracking_uri())
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "tensorflow_keras_example")
    print("Experiment Name:", experiment_name)
    
    run(alpha, run_origin, log_artifact)
