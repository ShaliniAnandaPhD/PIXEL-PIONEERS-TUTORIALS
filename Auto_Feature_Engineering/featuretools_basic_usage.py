# featuretools_basic_usage.py

"""
Tutorial: Basic Usage of Featuretools for Automated Feature Engineering
"""

# 1. Installation
# Ensure that Featuretools is installed
# You can install it using pip:
# pip install featuretools

# 2. Loading Required Libraries and Data
import featuretools as ft
import pandas as pd
import numpy as np

# Sample Data
# We'll use a simple mock dataset to demonstrate the functionality
data = {
    "customer_id": [1, 2, 3, 4, 5],
    "age": [34, 23, 45, 25, 32],
    "gender": ["M", "F", "F", "M", "F"],
    "region": ["North", "South", "East", "West", "North"],
    "spend": [100, 200, 300, 400, 500],
    "visit_date": pd.date_range(start="2023-01-01", periods=5, freq="D")
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# 3. Data Preparation
# Before creating features, we need to prepare our data
# Ensure there are no missing values and the data types are correct

df['customer_id'] = df['customer_id'].astype(str)
df['visit_date'] = pd.to_datetime(df['visit_date'])

# 4. Creating an EntitySet
# An EntitySet is a collection of entities (dataframes) and relationships between them

es = ft.EntitySet(id="customer_data")

# Adding the dataframe to the EntitySet
es = es.entity_from_dataframe(entity_id="customers",
                              dataframe=df,
                              index="customer_id",
                              time_index="visit_date")

print("EntitySet with customers entity:")
print(es)

# 5. Deep Feature Synthesis
# Deep Feature Synthesis (DFS) is the process of automatically generating features from the data

# Defining a dictionary for primitive options
primitive_options = {
    "aggregation_primitives": ["sum", "mean", "count"],
    "transform_primitives": ["day", "month", "year"]
}

# Running DFS
feature_matrix, feature_defs = ft.dfs(entityset=es,
                                      target_entity="customers",
                                      agg_primitives=primitive_options["aggregation_primitives"],
                                      trans_primitives=primitive_options["transform_primitives"])

print("Feature Matrix:")
print(feature_matrix)

# 6. Handling Time Variables
# Featuretools can handle time variables effectively

# Adding a new column with a timestamp
df['signup_date'] = pd.to_datetime(['2022-12-20', '2022-12-22', '2022-12-24', '2022-12-26', '2022-12-28'])

# Updating the EntitySet with the new column
es = es.entity_from_dataframe(entity_id="customers",
                              dataframe=df,
                              index="customer_id",
                              time_index="visit_date",
                              secondary_time_index={"signup_date": ['signup_date']})

print("EntitySet with secondary time index:")
print(es)

# 7. Advanced Aggregation and Transformation Primitives
# Using advanced primitives to generate more complex features

# Adding more aggregation and transformation primitives
advanced_primitive_options = {
    "aggregation_primitives": ["sum", "mean", "count", "min", "max", "std"],
    "transform_primitives": ["day", "month", "year", "weekday", "is_weekend"]
}

# Running DFS with advanced primitives
feature_matrix_advanced, feature_defs_advanced = ft.dfs(entityset=es,
                                                        target_entity="customers",
                                                        agg_primitives=advanced_primitive_options["aggregation_primitives"],
                                                        trans_primitives=advanced_primitive_options["transform_primitives"])

print("Advanced Feature Matrix:")
print(feature_matrix_advanced)

# 8. Saving and Loading Feature Definitions
# Saving the feature definitions to a file
feature_defs_file = "feature_defs.json"
ft.save_features(feature_defs_advanced, feature_defs_file)

# Loading the feature definitions from a file
loaded_feature_defs = ft.load_features(feature_defs_file)

# Verifying loaded feature definitions
print("Loaded Feature Definitions:")
print(loaded_feature_defs)

# 9. Visualization of Feature Definitions
# Featuretools provides tools to visualize the feature definitions

# Graphing feature dependencies
import featuretools.variable_types as vtypes
import featuretools.demo.load_mock_customer as load_mock_customer

graph = ft.graph_feature(feature_defs_advanced)
graph.view()

# 10. Conclusion
# Featuretools simplifies the process of feature engineering by automating the creation of meaningful features.
# This tutorial covered the basics of using Featuretools, including creating an EntitySet, using Deep Feature Synthesis,
# handling time variables, using advanced primitives, and saving/loading feature definitions.

print("Feature engineering with Featuretools complete.")
