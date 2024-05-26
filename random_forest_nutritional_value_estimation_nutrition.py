# File name: random_forest_nutritional_value_estimation_nutrition.py
# File library: Scikit-learn, Pandas, NumPy
# Use case: Nutrition - Nutritional Value Estimation

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Simulate meal data with nutritional values
np.random.seed(42)
num_meals = 1000
ingredients = ['apple', 'banana', 'spinach', 'carrot', 'chicken', 'beef', 'rice', 'pasta', 'bread', 'egg']
meals = [', '.join(np.random.choice(ingredients, size=np.random.randint(2, 6), replace=False)) for _ in range(num_meals)]
calories = np.random.randint(200, 800, size=num_meals)
protein = np.random.randint(10, 40, size=num_meals)
fat = np.random.randint(5, 30, size=num_meals)
carbohydrates = np.random.randint(20, 80, size=num_meals)

# Create a DataFrame with the simulated data
data = pd.DataFrame({
    'Meal': meals,
    'Calories': calories,
    'Protein': protein,
    'Fat': fat,
    'Carbohydrates': carbohydrates
})

# Preprocess the data
data['Ingredient_Count'] = data['Meal'].apply(lambda x: len(x.split(', ')))
data = pd.concat([data, pd.get_dummies(data['Meal'].str.join(', ').str.get_dummies(', '))], axis=1)
data.drop(['Meal'], axis=1, inplace=True)

# Split the data into features and target variables
X = data.drop(['Calories', 'Protein', 'Fat', 'Carbohydrates'], axis=1)
y_calories = data['Calories']
y_protein = data['Protein']
y_fat = data['Fat']
y_carbohydrates = data['Carbohydrates']

# Split the data into training and testing sets
X_train, X_test, y_calories_train, y_calories_test, y_protein_train, y_protein_test, y_fat_train, y_fat_test, y_carbohydrates_train, y_carbohydrates_test = train_test_split(
    X, y_calories, y_protein, y_fat, y_carbohydrates, test_size=0.2, random_state=42)

# Create and train the Random Forest models
rf_calories = RandomForestRegressor(n_estimators=100, random_state=42)
rf_calories.fit(X_train, y_calories_train)

rf_protein = RandomForestRegressor(n_estimators=100, random_state=42)
rf_protein.fit(X_train, y_protein_train)

rf_fat = RandomForestRegressor(n_estimators=100, random_state=42)
rf_fat.fit(X_train, y_fat_train)

rf_carbohydrates = RandomForestRegressor(n_estimators=100, random_state=42)
rf_carbohydrates.fit(X_train, y_carbohydrates_train)

# Make predictions on the test set
y_calories_pred = rf_calories.predict(X_test)
y_protein_pred = rf_protein.predict(X_test)
y_fat_pred = rf_fat.predict(X_test)
y_carbohydrates_pred = rf_carbohydrates.predict(X_test)

# Evaluate the models
print("Calories - Mean Absolute Error:", mean_absolute_error(y_calories_test, y_calories_pred))
print("Calories - Mean Squared Error:", mean_squared_error(y_calories_test, y_calories_pred))
print("Calories - R-squared:", r2_score(y_calories_test, y_calories_pred))

print("Protein - Mean Absolute Error:", mean_absolute_error(y_protein_test, y_protein_pred))
print("Protein - Mean Squared Error:", mean_squared_error(y_protein_test, y_protein_pred))
print("Protein - R-squared:", r2_score(y_protein_test, y_protein_pred))

print("Fat - Mean Absolute Error:", mean_absolute_error(y_fat_test, y_fat_pred))
print("Fat - Mean Squared Error:", mean_squared_error(y_fat_test, y_fat_pred))
print("Fat - R-squared:", r2_score(y_fat_test, y_fat_pred))

print("Carbohydrates - Mean Absolute Error:", mean_absolute_error(y_carbohydrates_test, y_carbohydrates_pred))
print("Carbohydrates - Mean Squared Error:", mean_squared_error(y_carbohydrates_test, y_carbohydrates_pred))
print("Carbohydrates - R-squared:", r2_score(y_carbohydrates_test, y_carbohydrates_pred))

# Simulate new meal data for prediction
new_meals = ['apple, banana, spinach, chicken', 'rice, beef, carrot', 'pasta, bread, egg']
new_data = pd.DataFrame({'Meal': new_meals})
new_data['Ingredient_Count'] = new_data['Meal'].apply(lambda x: len(x.split(', ')))
new_data = pd.concat([new_data, pd.get_dummies(new_data['Meal'].str.join(', ').str.get_dummies(', '))], axis=1)
new_data.drop(['Meal'], axis=1, inplace=True)

# Estimate nutritional values for new meals
new_calories = rf_calories.predict(new_data)
new_protein = rf_protein.predict(new_data)
new_fat = rf_fat.predict(new_data)
new_carbohydrates = rf_carbohydrates.predict(new_data)

# Print the estimated nutritional values for new meals
for i in range(len(new_meals)):
    print(f"Meal: {new_meals[i]}")
    print(f"  Estimated Calories: {new_calories[i]:.2f}")
    print(f"  Estimated Protein: {new_protein[i]:.2f}")
    print(f"  Estimated Fat: {new_fat[i]:.2f}")
    print(f"  Estimated Carbohydrates: {new_carbohydrates[i]:.2f}")
