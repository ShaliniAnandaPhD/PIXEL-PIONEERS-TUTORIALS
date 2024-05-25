# sklearn_water_quality_monitoring_environmental_science.py
# Scikit-learn, Pandas, Seaborn
# Use case: Environmental Science - Water Quality Monitoring

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Simulate water quality data
np.random.seed(42)
num_samples = 1000
pH = np.random.normal(loc=7.0, scale=0.5, size=num_samples)
turbidity = np.random.normal(loc=5.0, scale=2.0, size=num_samples)
chlorine = np.random.normal(loc=0.5, scale=0.2, size=num_samples)
lead = np.random.normal(loc=0.01, scale=0.005, size=num_samples)
quality = ['Good' if ph >= 6.5 and ph <= 8.5 and turb <= 5.0 and chlor >= 0.2 and chlor <= 1.0 and pb <= 0.015
           else 'Poor' for ph, turb, chlor, pb in zip(pH, turbidity, chlorine, lead)]

# Create a DataFrame with the simulated data
data = pd.DataFrame({
    'pH': pH,
    'Turbidity': turbidity,
    'Chlorine': chlorine,
    'Lead': lead,
    'Quality': quality
})

# Split the data into features and target
X = data.drop('Quality', axis=1)
y = data['Quality']

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

# Visualize the feature importance
importance = rf_classifier.feature_importances_
plt.figure(figsize=(8, 6))
sns.barplot(x=importance, y=X.columns)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()
