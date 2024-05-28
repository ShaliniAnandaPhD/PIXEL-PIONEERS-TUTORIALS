# File name: tensorflow_caloric_intake_prediction_nutrition.py
# File library: TensorFlow, Pandas, Scikit-learn
# Use case: Nutrition - Caloric Intake Prediction

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Simulate dietary data
np.random.seed(42)
num_samples = 1000
age = np.random.randint(18, 80, size=num_samples)
gender = np.random.choice(['Male', 'Female'], size=num_samples)
height = np.random.normal(loc=170, scale=10, size=num_samples)
weight = np.random.normal(loc=70, scale=15, size=num_samples)
activity_level = np.random.choice(['Sedentary', 'Moderately Active', 'Active'], size=num_samples)
caloric_intake = (
    (10 * weight) + (6.25 * height) - (5 * age) + (gender == 'Male') * 5 - (gender == 'Female') * 161 +
    (activity_level == 'Moderately Active') * 300 + (activity_level == 'Active') * 500 +
    np.random.normal(loc=0, scale=100, size=num_samples)
)

# Create a DataFrame with the simulated data
data = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'Height': height,
    'Weight': weight,
    'Activity Level': activity_level,
    'Caloric Intake': caloric_intake
})

# Convert categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data)

# Split the data into features and target
X = data.drop('Caloric Intake', axis=1)
y = data['Caloric Intake']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {loss:.4f}')
print(f'Test MAE: {mae:.4f}')

# Make predictions on new data
new_data = pd.DataFrame({
    'Age': [30],
    'Gender_Female': [1],
    'Gender_Male': [0],
    'Height': [165],
    'Weight': [60],
    'Activity Level_Active': [0],
    'Activity Level_Moderately Active': [1],
    'Activity Level_Sedentary': [0]
})
new_data_scaled = scaler.transform(new_data)
predicted_caloric_intake = model.predict(new_data_scaled)
print(f'Predicted Caloric Intake: {predicted_caloric_intake[0][0]:.2f} calories')

# Possible Errors and Solutions:

# ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type float).
# Solution: Ensure that the input data for the model is in the correct format. Use `.astype(np.float32)` to ensure the data type is compatible.

# AttributeError: 'numpy.ndarray' object has no attribute 'reshape'
# Solution: Check the shape of the input data and ensure it is a 2D array. Reshape the data if necessary using `.reshape(-1, 3)`.

# ValueError: Shapes (None, 1) and (None, 3) are incompatible
# Solution: Verify the input shape specified in the `input_shape` argument of the first Dense layer matches the shape of the training data.

# ModuleNotFoundError: No module named 'tensorflow'
# Solution: Ensure TensorFlow is installed using `pip install tensorflow`.

# IndexError: index 0 is out of bounds for axis 0 with size 0
# Solution: Check the data loading and preprocessing steps to ensure the data is correctly loaded and preprocessed before training.
