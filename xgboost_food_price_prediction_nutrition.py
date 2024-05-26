# File name: xgboost_food_price_prediction_nutrition.py
# File library: XGBoost, Pandas, Scikit-learn
# Use case: Nutrition - Food Price Prediction

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

# Simulate historical food price data
np.random.seed(42)
num_samples = 1000
food_items = ['Apple', 'Banana', 'Carrot', 'Spinach', 'Chicken', 'Beef', 'Salmon', 'Bread', 'Rice', 'Pasta']
dates = pd.date_range(start='2020-01-01', end='2022-12-31', periods=num_samples)
prices = np.random.uniform(low=1.0, high=10.0, size=num_samples)
quantities = np.random.randint(1, 10, size=num_samples)
promotions = np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])

# Create a DataFrame with the simulated data
data = pd.DataFrame({
    'Date': dates,
    'Food Item': np.random.choice(food_items, size=num_samples),
    'Price': prices,
    'Quantity': quantities,
    'Promotion': promotions
})

# Prepare the data for training
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data = pd.get_dummies(data, columns=['Food Item'])

# Split the data into features and target
X = data.drop(['Date', 'Price'], axis=1)
y = data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the XGBoost model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Predict prices for new data
new_data = pd.DataFrame({
    'Date': ['2023-06-15', '2023-07-01', '2023-07-20'],
    'Food Item': ['Apple', 'Chicken', 'Rice'],
    'Quantity': [5, 3, 2],
    'Promotion': [0, 1, 0]
})

new_data['Year'] = pd.to_datetime(new_data['Date']).dt.year
new_data['Month'] = pd.to_datetime(new_data['Date']).dt.month
new_data['Day'] = pd.to_datetime(new_data['Date']).dt.day
new_data = pd.get_dummies(new_data, columns=['Food Item'])

missing_columns = set(X.columns) - set(new_data.columns)
for column in missing_columns:
    new_data[column] = 0

new_data = new_data[X.columns]

predicted_prices = model.predict(new_data)
print("Predicted Prices:")
for i, price in enumerate(predicted_prices):
    print(f"{new_data['Date'][i]}: {price:.2f}")
