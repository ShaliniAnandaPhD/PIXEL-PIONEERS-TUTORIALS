# File name: xgboost_air_quality_forecasting_environmental_science.py
# File library: XGBoost, Pandas, Scikit-learn
# Use case: Environmental Science - Air Quality Forecasting

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Simulate air quality data
np.random.seed(42)
num_samples = 1000
pm25 = np.random.normal(loc=50, scale=20, size=num_samples)
no2 = np.random.normal(loc=30, scale=10, size=num_samples)
so2 = np.random.normal(loc=10, scale=5, size=num_samples)
co = np.random.normal(loc=5, scale=2, size=num_samples)
aqi = 0.4 * pm25 + 0.3 * no2 + 0.2 * so2 + 0.1 * co + np.random.normal(loc=0, scale=5, size=num_samples)

# Create a DataFrame with the simulated data
data = pd.DataFrame({
    'PM2.5': pm25,
    'NO2': no2,
    'SO2': so2,
    'CO': co,
    'AQI': aqi
})

# Split the data into features and target
X = data.drop('AQI', axis=1)
y = data['AQI']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an XGBoost regressor
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# Make predictions on new data
new_data = pd.DataFrame({
    'PM2.5': [60],
    'NO2': [35],
    'SO2': [12],
    'CO': [6]
})
aqi_forecast = model.predict(new_data)
print("Forecasted AQI:", aqi_forecast)

# Possible Errors and Solutions:

# KeyError: 'AQI'
# Solution: Ensure that the 'AQI' column exists in the data before processing. Use `data.head()` to check the column names.

# ValueError: could not convert string to float
# Solution: Check for any non-numeric values in the columns that are expected to be numeric. Use `data.info()` to inspect the data types.

# XGBoostError: feature_names mismatch
# Solution: Ensure that the training and new data have the same feature names and order. Verify that the preprocessing steps are applied consistently.

# IndexError: index out of bounds
# Solution: Ensure that the 'PM2.5', 'NO2', 'SO2', and 'CO' columns are present in `new_data` when making predictions. Use `new_data.columns` to check.

# ModuleNotFoundError: No module named 'xgboost'
# Solution: Ensure that the XGBoost library is installed. Use `pip install xgboost`.

