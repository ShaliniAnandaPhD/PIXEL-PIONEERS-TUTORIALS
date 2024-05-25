# File name: keras_forest_fire_prediction_environmental_science.py
# File library: Keras, Pandas, NumPy
# Use case: Environmental Science - Forest Fire Prediction

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Simulate forest fire data
np.random.seed(42)
num_samples = 1000
temperature = np.random.uniform(low=0, high=40, size=num_samples)
humidity = np.random.uniform(low=0, high=100, size=num_samples)
wind_speed = np.random.uniform(low=0, high=30, size=num_samples)
rainfall = np.random.uniform(low=0, high=10, size=num_samples)
fire_occurrence = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])

# Create a DataFrame with the simulated data
data = pd.DataFrame({
    'Temperature': temperature,
    'Humidity': humidity,
    'Wind Speed': wind_speed,
    'Rainfall': rainfall,
    'Fire Occurrence': fire_occurrence
})

# Split the data into features and target
X = data.drop('Fire Occurrence', axis=1)
y = data['Fire Occurrence']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the Keras model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(4,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Make predictions on new data
new_data = np.array([[25, 60, 10, 2]])
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print("Prediction:", prediction)
