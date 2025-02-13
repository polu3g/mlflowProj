import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load the dataset
file_path = 'Cartridge_Promocode.csv'
try:
    data = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv(file_path, encoding='latin1')

# Display the first few rows
print(data.head())

# Separate features and target
X = data["Cartridge"].values.reshape(-1, 1)
y = data["Promocode"].values

# Encode categorical features using OneHotEncoder
encoder_X = OneHotEncoder(sparse_output=False)
X_encoded = encoder_X.fit_transform(X)

# Encode target labels using OneHotEncoder
encoder_y = OneHotEncoder(sparse_output=False)
y_encoded = encoder_y.fit_transform(y.reshape(-1, 1))

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(y_train.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train the model
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=16,
                    validation_split=0.1,
                    callbacks=[early_stopping, reduce_lr],
                    verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Save the model
model_save_path = 'promo_code_model_final.h5'
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Load the model
loaded_model = tf.keras.models.load_model(model_save_path)
print("Model loaded successfully.")

# Make predictions
new_cartridge = "HP 920 (Cyan)"
new_cartridge_encoded = encoder_X.transform([[new_cartridge]])
new_cartridge_scaled = scaler.transform(new_cartridge_encoded)
prediction = loaded_model.predict(new_cartridge_scaled)
predicted_class = np.argmax(prediction)
decoded_prediction = encoder_y.inverse_transform([[1 if i == predicted_class else 0 for i in range(len(prediction[0]))]])
print(f"Predicted Promocode for cartridge '{new_cartridge}': {decoded_prediction[0][0]}")
