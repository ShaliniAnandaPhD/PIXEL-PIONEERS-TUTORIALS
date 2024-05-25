# tensorflow_smart_irrigation_system_agriculture.py
# File library: TensorFlow, Pandas, Matplotlib
# Use case: Agriculture - Smart Irrigation System

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Simulate soil moisture data
np.random.seed(42)
num_samples = 1000
soil_moisture = np.random.uniform(low=0, high=100, size=num_samples)
temperature = np.random.uniform(low=10, high=40, size=num_samples)
humidity = np.random.uniform(low=20, high=80, size=num_samples)
irrigation_needed = np.where((soil_moisture < 30) & (temperature > 25) & (humidity < 50), 1, 0)

# Create a DataFrame with the simulated data
data = pd.DataFrame({
    'Soil Moisture': soil_moisture,
    'Temperature': temperature,
    'Humidity': humidity,
    'Irrigation Needed': irrigation_needed
})

# Split the data into features and target
X = data.drop('Irrigation Needed', axis=1)
y = data['Irrigation Needed']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Plot the training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Make predictions on new data
new_data = np.array([[25, 30, 60]])  # Soil moisture, temperature, humidity
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print("Irrigation Needed:", prediction)
