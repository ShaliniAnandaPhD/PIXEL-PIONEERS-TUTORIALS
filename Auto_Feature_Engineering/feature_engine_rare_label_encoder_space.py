# feature_engine_rare_label_encoder_space.py

"""
Tutorial: Handling Rare Labels in Categorical Data with Feature Engine for Space Exploration Data

This tutorial covers:
1. Installation
2. Loading Required Libraries and Data
3. Data Preparation
4. Identifying Rare Labels in Space Exploration Data
5. Handling Rare Labels with Feature Engine's RareLabelEncoder
6. Combining RareLabelEncoder with Other Feature Engineering Steps
7. Advanced Usage and Customization
8. Error Handling and Troubleshooting
9. Conclusion
"""

# 1. Installation
# Ensure that Feature Engine is installed
# You can install it using pip:
# pip install feature-engine

# 2. Loading Required Libraries and Data
import pandas as pd
from feature_engine.encoding import RareLabelEncoder

# Sample Space Exploration Data
# Creating a mock dataset with categorical variables to demonstrate the functionality
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
        "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", 
        "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", 
        "Yes", "Yes", "Yes", "No", "Yes", "Yes", "Yes"
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

# 4. Identifying Rare Labels in Space Exploration Data
# Before handling rare labels, it's important to identify them
label_counts = df['spacecraft'].value_counts()
print("\nLabel Counts:")
print(label_counts)

# Defining a threshold for rare labels (e.g., categories that appear less than 2 times are considered rare)
threshold = 2
rare_labels = label_counts[label_counts < threshold].index.tolist()
print("\nRare Labels:")
print(rare_labels)

# 5. Handling Rare Labels with Feature Engine's RareLabelEncoder
# Using Feature Engine's RareLabelEncoder to handle rare labels

# Initializing the encoder
rare_label_encoder = RareLabelEncoder(tol=0.05, n_categories=1, replace_with='Rare')

# Fitting the encoder
df_encoded = rare_label_encoder.fit_transform(df)

print("\nEncoded DataFrame:")
print(df_encoded)

# 6. Combining RareLabelEncoder with Other Feature Engineering Steps
# Combining RareLabelEncoder with other encoders or transformers

from feature_engine.encoding import OneHotEncoder

# Combining RareLabelEncoder with OneHotEncoder
# First, apply the RareLabelEncoder
df_encoded = rare_label_encoder.fit_transform(df)

# Then, apply OneHotEncoder
one_hot_encoder = OneHotEncoder()
df_encoded = one_hot_encoder.fit_transform(df_encoded)

print("\nOne-Hot Encoded DataFrame after Handling Rare Labels:")
print(df_encoded)

# 7. Advanced Usage and Customization
# Customizing RareLabelEncoder for more advanced usage

# Setting a different threshold for rare labels
rare_label_encoder_custom = RareLabelEncoder(tol=0.10, n_categories=1, replace_with='Other')

# Fitting the encoder
df_encoded_custom = rare_label_encoder_custom.fit_transform(df)

print("\nCustom Encoded DataFrame:")
print(df_encoded_custom)

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
        "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", 
        "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", 
        "Yes", "Yes", "Yes", "No", "Yes", "Yes", "Yes"
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

# Applying RareLabelEncoder to multiple columns
rare_label_encoder_multi = RareLabelEncoder(tol=0.05, n_categories=1, variables=['spacecraft', 'launch_site'], replace_with='Rare')

# Fitting the encoder
df_multi_encoded = rare_label_encoder_multi.fit_transform(df_multi)

print("\nMulti-Column Encoded DataFrame:")
print(df_multi_encoded)

# 8. Error Handling and Troubleshooting
# Discussing potential errors and their solutions

# Potential Error: Invalid column name
# Solution: Ensure the column names provided to the encoder exist in the DataFrame

try:
    # Example: Invalid column name
    rare_label_encoder_error = RareLabelEncoder(tol=0.05, n_categories=1, variables=['nonexistent_column'], replace_with='Rare')
    df_error = rare_label_encoder_error.fit_transform(df)
except Exception as e:
    print("\nError encountered due to invalid column name:")
    print(e)

# Potential Error: Incorrect data types
# Solution: Ensure all columns to be encoded are of type string or categorical

try:
    # Example: Incorrect data type for a column
    df_error = df.copy()
    df_error['cost_million_usd'] = df_error['cost_million_usd'].astype(str)
    rare_label_encoder_error = RareLabelEncoder(tol=0.05, n_categories=1, variables=['cost_million_usd'], replace_with='Rare')
    df_error_encoded = rare_label_encoder_error.fit_transform(df_error)
except Exception as e:
   Here is a continuation and completion of the tutorial:

