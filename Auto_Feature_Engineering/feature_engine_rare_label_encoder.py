# feature_engine_rare_label_encoder.py

"""
Tutorial: Handling Rare Labels in Categorical Data with Feature Engine

This tutorial covers:
1. Installation
2. Loading Required Libraries and Data
3. Data Preparation
4. Identifying Rare Labels
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

# Sample Data
# Creating a mock dataset with categorical variables to demonstrate the functionality
data = {
    "category": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "A", "B", "C", "A", "B", "D", "E", "A", "A", "A"],
    "value": [10, 15, 10, 20, 30, 25, 15, 10, 5, 10, 15, 10, 20, 30, 25, 15, 10, 5, 10, 15]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# 3. Data Preparation
# Ensuring data types are correct and handling missing values if any
df['category'] = df['category'].astype(str)

# 4. Identifying Rare Labels
# Before handling rare labels, it's important to identify them
label_counts = df['category'].value_counts()
print("\nLabel Counts:")
print(label_counts)

# Defining a threshold for rare labels (e.g., categories that appear less than 3 times are considered rare)
threshold = 3
rare_labels = label_counts[label_counts < threshold].index.tolist()
print("\nRare Labels:")
print(rare_labels)

# 5. Handling Rare Labels with Feature Engine's RareLabelEncoder
# Using Feature Engine's RareLabelEncoder to handle rare labels

# Initializing the encoder
rare_label_encoder = RareLabelEncoder(tol=0.1, n_categories=1, replace_with='Rare')

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
rare_label_encoder_custom = RareLabelEncoder(tol=0.05, n_categories=2, replace_with='Other')

# Fitting the encoder
df_encoded_custom = rare_label_encoder_custom.fit_transform(df)

print("\nCustom Encoded DataFrame:")
print(df_encoded_custom)

# Handling multiple categorical columns
data_multi = {
    "category1": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "A", "B", "C", "A", "B", "D", "E", "A", "A", "A"],
    "category2": ["X", "Y", "Z", "X", "Y", "Z", "X", "Y", "Z", "X", "Y", "Z", "X", "Y", "Z", "X", "Y", "Z", "X", "Y"],
    "value": [10, 15, 10, 20, 30, 25, 15, 10, 5, 10, 15, 10, 20, 30, 25, 15, 10, 5, 10, 15]
}

df_multi = pd.DataFrame(data_multi)
df_multi['category1'] = df_multi['category1'].astype(str)
df_multi['category2'] = df_multi['category2'].astype(str)

# Applying RareLabelEncoder to multiple columns
rare_label_encoder_multi = RareLabelEncoder(tol=0.1, n_categories=1, variables=['category1', 'category2'], replace_with='Rare')

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
    rare_label_encoder_error = RareLabelEncoder(tol=0.1, n_categories=1, variables=['nonexistent_column'], replace_with='Rare')
    df_error = rare_label_encoder_error.fit_transform(df)
except Exception as e:
    print("\nError encountered due to invalid column name:")
    print(e)

# Potential Error: Incorrect data types
# Solution: Ensure all columns to be encoded are of type string or categorical

try:
    # Example: Incorrect data type for a column
    df_error = df.copy()
    df_error['value'] = df_error['value'].astype(str)
    rare_label_encoder_error = RareLabelEncoder(tol=0.1, n_categories=1, variables=['value'], replace_with='Rare')
    df_error_encoded = rare_label_encoder_error.fit_transform(df_error)
except Exception as e:
    print("\nError encountered due to incorrect data type:")
    print(e)

# 9. Conclusion
# Feature Engine simplifies the process of handling rare labels in categorical data by providing an easy-to-use RareLabelEncoder.
# This tutorial covered the basics of using RareLabelEncoder, combining it with other feature engineering steps,
# customizing the encoder for advanced usage, and handling potential errors.

print("Handling rare labels in categorical data with Feature Engine complete.")
