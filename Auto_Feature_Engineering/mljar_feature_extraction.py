# mljar_feature_extraction_space.py

"""
Tutorial: Automatic Feature Extraction and Selection using MLJAR for Space Exploration Data

This tutorial covers:
1. Installation
2. Loading Required Libraries and Data
3. Data Preparation
4. Automatic Feature Extraction with MLJAR
5. Feature Selection with MLJAR
6. Combining Feature Extraction and Selection
7. Advanced Usage and Customization
8. Error Handling and Troubleshooting
9. Conclusion
"""

# 1. Installation
# Ensure that MLJAR is installed
# You can install it using pip:
# pip install mljar-supervised

# 2. Loading Required Libraries and Data
import pandas as pd
from supervised.automl import AutoML

# Sample Space Exploration Data
# Creating a mock dataset to demonstrate the functionality
data = {
    "mission": [
        "Apollo 11", "Voyager 1", "Voyager 2", "Mars Rover", "Hubble", "Galileo", "Chandra", 
        "Cassini", "New Horizons", "Juno", "Pioneer 10", "Pioneer 11", "Kepler", 
        "ISS", "Sputnik", "Apollo 12", "Apollo 13", "Apollo 14", "Apollo 15", "Apollo 16"
    ],
    "spacecraft": [
        "Saturn V", "Titan IIIE", "Titan IIIE", "Delta II", "STS-31", "STS-34", "STS-93", 
        "Titan IVB", "Atlas V", "Atlas V", "Atlas-Centaur", "Atlas-Centaur", "Delta II", 
        "Soyuz", "R-7", "Saturn V", "Saturn V", "Saturn V", "Saturn V", "Saturn V"
    ],
    "success": [
        1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 
        1, 1, 1, 0, 1, 1, 1
    ],
    "year": [
        1969, 1977, 1977, 2003, 1990, 1989, 1999, 
        1997, 2006, 2011, 1972, 1973, 2009, 
        1998, 1957, 1969, 1970, 1971, 1971, 1972
    ],
    "cost_million_usd": [
        25, 45, 45, 800, 2500, 1000, 1500, 
        3500, 700, 1200, 370, 390, 600, 
        150000, 300, 25, 25, 25, 25, 25
    ]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# 3. Data Preparation
# Ensuring data types are correct and handling missing values if any
df['spacecraft'] = df['spacecraft'].astype(str)
df['success'] = df['success'].astype(int)

# Splitting the data into features and target
X = df.drop(columns=['success'])
y = df['success']

# 4. Automatic Feature Extraction with MLJAR
# Using MLJAR for automatic feature extraction

# Initializing the AutoML object
automl = AutoML(mode="Explain", total_time_limit=600)

# Fitting the AutoML object
automl.fit(X, y)

# Extracting the feature importance
importance = automl.report()['importance']
print("\nFeature Importance:")
print(importance)

# 5. Feature Selection with MLJAR
# Using MLJAR for automatic feature selection

# Selecting the most important features
selected_features = importance[importance['importance'] > 0.01]['feature'].tolist()
print("\nSelected Features:")
print(selected_features)

# 6. Combining Feature Extraction and Selection
# Combining feature extraction and selection steps

X_selected = X[selected_features]

# Re-running AutoML on the selected features
automl_selected = AutoML(mode="Explain", total_time_limit=300)
automl_selected.fit(X_selected, y)

# 7. Advanced Usage and Customization
# Customizing MLJAR for more advanced usage

# Setting a different mode for AutoML
automl_advanced = AutoML(mode="Perform", total_time_limit=600, explain_level=2)
automl_advanced.fit(X, y)

# Extracting the feature importance for the advanced run
importance_advanced = automl_advanced.report()['importance']
print("\nAdvanced Feature Importance:")
print(importance_advanced)

# Handling multiple categorical columns
data_multi = {
    "mission": [
        "Apollo 11", "Voyager 1", "Voyager 2", "Mars Rover", "Hubble", "Galileo", "Chandra", 
        "Cassini", "New Horizons", "Juno", "Pioneer 10", "Pioneer 11", "Kepler", 
        "ISS", "Sputnik", "Apollo 12", "Apollo 13", "Apollo 14", "Apollo 15", "Apollo 16"
    ],
    "spacecraft": [
        "Saturn V", "Titan IIIE", "Titan IIIE", "Delta II", "STS-31", "STS-34", "STS-93", 
        "Titan IVB", "Atlas V", "Atlas V", "Atlas-Centaur", "Atlas-Centaur", "Delta II", 
        "Soyuz", "R-7", "Saturn V", "Saturn V", "Saturn V", "Saturn V", "Saturn V"
    ],
    "launch_site": [
        "KSC", "KSC", "KSC", "CCAFS", "KSC", "KSC", "KSC", 
        "CCAFS", "CCAFS", "CCAFS", "KSC", "KSC", "CCAFS", 
        "Baikonur", "Baikonur", "KSC", "KSC", "KSC", "KSC", "KSC"
    ],
    "success": [
        1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 
        1, 1, 1, 0, 1, 1, 1
    ],
    "year": [
        1969, 1977, 1977, 2003, 1990, 1989, 1999, 
        1997, 2006, 2011, 1972, 1973, 2009, 
        1998, 1957, 1969, 1970, 1971, 1971, 1972
    ],
    "cost_million_usd": [
        25, 45, 45, 800, 2500, 1000, 1500, 
        3500, 700, 1200, 370, 390, 600, 
        150000, 300, 25, 25, 25, 25, 25
    ]
}

df_multi = pd.DataFrame(data_multi)
df_multi['spacecraft'] = df_multi['spacecraft'].astype(str)
df_multi['launch_site'] = df_multi['launch_site'].astype(str)
df_multi['success'] = df_multi['success'].astype(int)

# Splitting the data into features and target
X_multi = df_multi.drop(columns=['success'])
y_multi = df_multi['success']

# Running AutoML on the multi-column data
automl_multi = AutoML(mode="Explain", total_time_limit=600)
automl_multi.fit(X_multi, y_multi)

# Extracting the feature importance for the multi-column data
importance_multi = automl_multi.report()['importance']
print("\nMulti-Column Feature Importance:")
print(importance_multi)

# 8. Error Handling and Troubleshooting
# Discussing potential errors and their solutions

# Potential Error: Invalid data format
# Solution: Ensure the data format is correct before running AutoML

try:
    # Example: Invalid data format
    X_error = X.copy()
    X_error['year'] = X_error['year'].astype(str)
    automl_error = AutoML(mode="Explain", total_time_limit=600)
    automl_error.fit(X_error, y)
except Exception as e:
    print("\nError encountered due to invalid data format:")
    print(e)

# Potential Error: Missing data in key columns
# Solution: Fill missing values or drop rows with missing data

try:
    # Example: Missing data in key columns
    X_error = X.copy()
    X_error.loc[0, 'year'] = None
    automl_error = AutoML(mode="Explain", total_time_limit=600)
    automl_error.fit(X_error, y)
except Exception as e:
    print("\nError encountered due to missing data:")
    print(e)

# 9. Conclusion
# MLJAR simplifies the process of automatic feature extraction and selection for complex datasets.
# This tutorial covered the basics of using MLJAR for feature extraction and selection,Here is a detailed tutorial on automatic feature extraction and selection using MLJAR with a use case involving space exploration data. The file is named `mljar_feature_extraction_space.py`.
