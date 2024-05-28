
# Use case: Renewable Energy

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Simulate the dataset
np.random.seed(42)
num_samples = 1000
temperature = np.random.uniform(low=10, high=35, size=num_samples)
humidity = np.random.uniform(low=0.2, high=0.9, size=num_samples)
wind_speed = np.random.uniform(low=0, high=10, size=num_samples)
cloud_cover = np.random.uniform(low=0, high=1, size=num_samples)
solar_energy = (temperature * 0.8) + (humidity * 0.6) - (cloud_cover * 0.9) + (wind_speed * 0.4) + np.random.normal(0, 2, size=num_samples)

data = pd.DataFrame({
    'Temperature': temperature,
    'Humidity': humidity,
    'Wind_Speed': wind_speed,
    'Cloud_Cover': cloud_cover,
    'Solar_Energy': solar_energy
})

# Split the data into features and target
X = data.drop(['Solar_Energy'], axis=1)  # Features (weather data)
y = data['Solar_Energy']  # Target variable (solar energy output)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
print(f'Test loss: {test_loss:.4f}')
print(f'Test MAE: {test_mae:.4f}')

# Make predictions on new data
new_data = pd.DataFrame({
    'Temperature': [25.0],
    'Humidity': [0.6],
    'Wind_Speed': [5.0],
    'Cloud_Cover': [0.3]
})
new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)
print(f'Predicted solar energy output: {predictions[0][0]:.4f}')
