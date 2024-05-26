# File name: health_insurance_fraud_detection_machine_learning.py
# File library: Scikit-learn, Pandas
# Use case: Healthcare - Health Insurance Fraud Detection

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Generate synthetic health insurance claim data
num_claims = 10000
claim_id = np.arange(1, num_claims + 1)
age = np.random.randint(18, 80, size=num_claims)
gender = np.random.choice(['Male', 'Female'], size=num_claims)
bmi = np.random.normal(25, 5, size=num_claims)
num_children = np.random.randint(0, 5, size=num_claims)
smoker = np.random.choice(['Yes', 'No'], size=num_claims)
region = np.random.choice(['Northeast', 'Northwest', 'Southeast', 'Southwest'], size=num_claims)
charges = np.random.normal(10000, 5000, size=num_claims)
fraud = np.random.choice([0, 1], size=num_claims, p=[0.95, 0.05])

# Create a DataFrame with the synthetic data
data = pd.DataFrame({
    'Claim_ID': claim_id,
    'Age': age,
    'Gender': gender,
    'BMI': bmi,
    'Num_Children': num_children,
    'Smoker': smoker,
    'Region': region,
    'Charges': charges,
    'Fraud': fraud
})

# Preprocess the data
data = pd.get_dummies(data, columns=['Gender', 'Smoker', 'Region'])

# Split the data into features and target
X = data.drop(['Claim_ID', 'Fraud'], axis=1)
y = data['Fraud']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Predict fraud for new claims
new_claims = pd.DataFrame({
    'Age': [35, 42, 58],
    'Gender_Female': [1, 0, 1],
    'Gender_Male': [0, 1, 0],
    'BMI': [28.5, 24.2, 31.7],
    'Num_Children': [2, 0, 1],
    'Smoker_No': [1, 0, 1],
    'Smoker_Yes': [0, 1, 0],
    'Region_Northeast': [0, 0, 1],
    'Region_Northwest': [1, 0, 0],
    'Region_Southeast': [0, 1, 0],
    'Region_Southwest': [0, 0, 0],
    'Charges': [8500, 12000, 9800]
})

fraud_predictions = rf_classifier.predict(new_claims)
print("Fraud Predictions:")
print(fraud_predictions)
