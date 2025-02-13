import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, LSTM, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import mlflow
import mlflow.keras
import pickle

# Load and clean data
file_path = "Cartridge_list.csv"  # Replace with your file path
data = pd.read_csv(file_path, encoding="ISO-8859-1")

# Clean and preprocess
data['Compatible Printers'] = data['Compatible Printers'].str.replace(r'[^\w\s,]', '', regex=True)
data['Alternative'].fillna('No Alternative', inplace=True)

# Combine Cartridge and Alternative for prediction target
data['Target'] = data['Cartridge'] + " | " + data['Alternative']

# Tokenize inputs
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['Compatible Printers'])
X = tokenizer.texts_to_sequences(data['Compatible Printers'])
X = pad_sequences(X, padding='post')

# Encode targets
target_encoder = LabelEncoder()
data['Target_Encoded'] = target_encoder.fit_transform(data['Target'])
y = data['Target_Encoded']

# Save tokenizer and encoder
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("target_encoder.pkl", "wb") as f:
    pickle.dump(target_encoder, f)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=X.shape[1]),
    Bidirectional(LSTM(128, return_sequences=True)),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(target_encoder.classes_), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)

# Log to MLflow
with mlflow.start_run():
    # Train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=20, batch_size=16, callbacks=[early_stopping])

    # Log metrics and model
    for epoch in range(len(history.history['accuracy'])):
        mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
        mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
        mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)

    # Log model
    mlflow.keras.log_model(model, "lp_model")
    registered_model = mlflow.register_model("runs:/{}/lp_model".format(mlflow.active_run().info.run_id), "lp_model")


# Prediction function with unseen label handling
def predict(printer_name):
    # Load tokenizer and encoder
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("target_encoder.pkl", "rb") as f:
        target_encoder = pickle.load(f)

    # Tokenize input
    printer_sequence = tokenizer.texts_to_sequences([printer_name])
    printer_sequence = pad_sequences(printer_sequence, maxlen=X.shape[1], padding='post')

    # Load the model from MLflow Model Registry
    model_uri = f'models:/{registered_model.name}/{registered_model.version}'
    loaded_model = mlflow.keras.load_model(model_uri)

    # Predict
    predictions = loaded_model.predict(printer_sequence)
    predicted_index = np.argmax(predictions)
    return target_encoder.inverse_transform([predicted_index])[0]


# Example usage
printer_name = "HP Color LaserJet 1502"
result = predict(printer_name)
print(f"Compatible Cartridge and Alternative for {printer_name}: {result}")
