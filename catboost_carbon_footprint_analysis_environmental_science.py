import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Simulate carbon footprint data
np.random.seed(42)
num_samples = 1000
transportation = np.random.choice(['Car', 'Public Transport', 'Bicycle', 'Walking'], size=num_samples)
diet = np.random.choice(['Meat Eater', 'Vegetarian', 'Vegan'], size=num_samples)
energy_usage = np.random.uniform(low=10, high=100, size=num_samples)
waste_production = np.random.uniform(low=0.5, high=2.0, size=num_samples)
carbon_footprint = (
    (transportation == 'Car') * np.random.uniform(low=8, high=12, size=num_samples) +
    (transportation == 'Public Transport') * np.random.uniform(low=2, high=4, size=num_samples) +
    (transportation == 'Bicycle') * np.random.uniform(low=0.1, high=0.3, size=num_samples) +
    (transportation == 'Walking') * np.random.uniform(low=0, high=0.1, size=num_samples) +
    (diet == 'Meat Eater') * np.random.uniform(low=3, high=5, size=num_samples) +
    (diet == 'Vegetarian') * np.random.uniform(low=1.5, high=2.5, size=num_samples) +
    (diet == 'Vegan') * np.random.uniform(low=1, high=2, size=num_samples) +
    energy_usage * 0.1 + waste_production * 0.5 + np.random.normal(loc=0, scale=1, size=num_samples)
)

# Create a DataFrame with the simulated data
data = pd.DataFrame({
    'Transportation': transportation,
    'Diet': diet,
    'Energy Usage': energy_usage,
    'Waste Production': waste_production,
    'Carbon Footprint': carbon_footprint
})

# Split the data into features and target
X = data.drop('Carbon Footprint', axis=1)
y = data['Carbon Footprint']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the CatBoost model
model = CatBoostRegressor(iterations=100, learning_rate=0.1, random_seed=42)
model.fit(X_train, y_train, cat_features=['Transportation', 'Diet'], eval_set=(X_test, y_test), plot=True)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

# Visualize the feature importance
feature_importance = model.get_feature_importance(prettified=True)
sns.barplot(x='Importances', y='Feature Id', data=feature_importance)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# Predict carbon footprint for new data
new_data = pd.DataFrame({
    'Transportation': ['Car', 'Public Transport', 'Bicycle'],
    'Diet': ['Meat Eater', 'Vegetarian', 'Vegan'],
    'Energy Usage': [80, 50, 30],
    'Waste Production': [1.5, 1.0, 0.8]
})
new_predictions = model.predict(new_data)
print("Carbon Footprint Predictions for New Data:")
print(new_predictions)

# Possible Errors and Solutions:
# 1. Import Errors:
#    Error: "ModuleNotFoundError: No module named 'catboost'"
#    Solution: Ensure CatBoost and other required libraries are properly installed. Use `pip install catboost seaborn`.

# 2. Data Type Errors:
#    Error: "ValueError: could not convert string to float"
#    Solution: Ensure categorical features are properly specified in CatBoost using the `cat_features` parameter.

# 3. Shape Mismatch Errors:
#    Error: "ValueError: shapes (X,Y) and (Y,Z) not aligned"
#    Solution: Verify that the input features and target labels have the correct shapes and alignments.

# 4. Evaluation Errors:
#    Error: "ValueError: Found input variables with inconsistent numbers of samples"
#    Solution: Ensure that the training and testing data have consistent sample sizes and are correctly split.

# 5. Performance Issues:
#    Solution: Use smaller datasets or fewer iterations if the training process is too slow. Consider using GPU acceleration by installing CatBoost with GPU support.
