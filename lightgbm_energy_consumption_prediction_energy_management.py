
# Use case: Energy Management - Energy Consumption Prediction

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Simulate energy consumption data
np.random.seed(42)
num_samples = 1000
num_households = 100
household_ids = np.arange(1, num_households + 1)
dates = pd.date_range(start='2022-01-01', periods=num_samples // num_households)
energy_consumption = np.random.uniform(low=0, high=50, size=num_samples)
temperature = np.random.uniform(low=10, high=40, size=num_samples)
humidity = np.random.uniform(low=20, high=80, size=num_samples)

# Create a DataFrame with the simulated data
data = pd.DataFrame({
    'Household ID': np.repeat(household_ids, num_samples // num_households),
    'Date': np.tile(dates, num_households),
    'Energy Consumption': energy_consumption,
    'Temperature': temperature,
    'Humidity': humidity
})

# Split the data into features and target
X = data.drop('Energy Consumption', axis=1)
y = data['Energy Consumption']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM dataset objects
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Set LightGBM parameters
params = {
    'objective': 'regression',
    'metric': 'mean_squared_error',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# Train the LightGBM model
model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[test_data], early_stopping_rounds=10)

# Make predictions on the test set
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# Make predictions on new data
new_data = pd.DataFrame({
    'Household ID': [101, 102, 103],
    'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'Temperature': [25, 30, 28],
    'Humidity': [60, 70, 65]
})
new_predictions = model.predict(new_data)
print("Predictions for new data:")
print(new_predictions)
