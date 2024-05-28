# sklearn_water_quality_monitoring_environmental_science.py
# Libraries: Scikit-learn, Pandas, Seaborn
# Use case: Environmental Science - Water Quality Monitoring

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Simulate water quality data
# Set the random seed for reproducibility
np.random.seed(42)

# Number of samples in the dataset
num_samples = 1000

# Simulate data for each water quality parameter
pH = np.random.normal(loc=7.0, scale=0.5, size=num_samples)
turbidity = np.random.normal(loc=5.0, scale=2.0, size=num_samples)
chlorine = np.random.normal(loc=0.5, scale=0.2, size=num_samples)
lead = np.random.normal(loc=0.01, scale=0.005, size=num_samples)

# Generate quality labels based on some criteria
# Here, we define "Good" quality based on arbitrary threshold values
quality = ['Good' if 6.5 <= ph <= 8.5 and turb <= 5.0 and 0.2 <= chlor <= 1.0 and pb <= 0.015 else 'Poor'
           for ph, turb, chlor, pb in zip(pH, turbidity, chlorine, lead)]

# Create a DataFrame with the simulated data
data = pd.DataFrame({
    'pH': pH,
    'Turbidity': turbidity,
    'Chlorine': chlorine,
    'Lead': lead,
    'Quality': quality
})

# Display the first few rows of the DataFrame to verify the data
print(data.head())

# Split the data into features and target

X = data.drop('Quality', axis=1)  # Features
y = data['Quality']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set

y_pred = rf_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Possible Errors and Solutions:
# 1. ValueError: Input contains NaN, infinity or a value too large for dtype('float32').
#    Solution: Ensure that the input data does not contain any NaN or infinite values. You can use data.isnull().sum() to check for NaNs.

# 2. ValueError: Number of labels=1 does not match number of samples.
#    Solution: This occurs if there is only one class in the training set. Make sure your target variable has at least two classes.

# Visualize the feature importance
importance = rf_classifier.feature_importances_

# Create a bar plot for feature importance
plt.figure(figsize=(8, 6))
sns.barplot(x=importance, y=X.columns)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# Additional Details:
# - RandomForestClassifier is an ensemble learning method based on constructing multiple decision trees and combining their outputs.
# - The feature importance plot helps to understand which features contribute most to the prediction.
# - Make sure to analyze the classification report, which provides precision, recall, and F1-score for each class, giving a better understanding of model performance.

