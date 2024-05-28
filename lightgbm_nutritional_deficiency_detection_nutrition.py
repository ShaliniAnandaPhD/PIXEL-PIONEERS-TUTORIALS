
# Use case: Nutrition - Nutritional Deficiency Detection

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb

# Simulate nutritional deficiency data
np.random.seed(42)
num_samples = 1000
ages = np.random.randint(18, 80, size=num_samples)
genders = np.random.choice(['Male', 'Female'], size=num_samples)
vitaminA_levels = np.random.normal(loc=50, scale=20, size=num_samples)
vitaminD_levels = np.random.normal(loc=30, scale=10, size=num_samples)
calcium_levels = np.random.normal(loc=9.5, scale=1.5, size=num_samples)
iron_levels = np.random.normal(loc=120, scale=30, size=num_samples)

deficiencies = []
for i in range(num_samples):
    deficiency = []
    if vitaminA_levels[i] < 30:
        deficiency.append('Vitamin A')
    if vitaminD_levels[i] < 20:
        deficiency.append('Vitamin D')
    if calcium_levels[i] < 8.5:
        deficiency.append('Calcium')
    if iron_levels[i] < 60:
        deficiency.append('Iron')
    if not deficiency:
        deficiency.append('None')
    deficiencies.append(', '.join(deficiency))

# Create a DataFrame with the simulated data
data = pd.DataFrame({
    'Age': ages,
    'Gender': genders,
    'Vitamin A': vitaminA_levels,
    'Vitamin D': vitaminD_levels,
    'Calcium': calcium_levels,
    'Iron': iron_levels,
    'Deficiency': deficiencies
})

# Preprocess the data
data = pd.get_dummies(data, columns=['Gender'])
data['Deficiency'] = data['Deficiency'].apply(lambda x: x if x != 'None' else 'No Deficiency')

# Split the data into features and target
X = data.drop('Deficiency', axis=1)
y = data['Deficiency']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the LightGBM model
model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# Simulate new patient data
new_patients = pd.DataFrame({
    'Age': [35, 52, 68],
    'Gender_Female': [1, 0, 1],
    'Gender_Male': [0, 1, 0],
    'Vitamin A': [45, 60, 25],
    'Vitamin D': [25, 18, 32],
    'Calcium': [9.2, 8.8, 7.9],
    'Iron': [110, 90, 70]
})

# Predict deficiencies for new patients
predicted_deficiencies = model.predict(new_patients)

# Print the predicted deficiencies
for i, deficiency in enumerate(predicted_deficiencies):
    print(f"Patient {i+1}: {deficiency}")
