# featuretools_deep_feature_synthesis.py

"""
Tutorial: Deep Feature Synthesis with Featuretools for Automated Feature Engineering

This tutorial covers:
1. Installation
2. Loading Required Libraries and Data
3. Data Preparation
4. Creating an EntitySet
5. Deep Feature Synthesis with Multiple Tables
6. Handling Relationships
7. Advanced Aggregation and Transformation Primitives
8. Custom Primitives
9. Handling Time Variables and Time-Based Features
10. Error Handling and Troubleshooting
11. Conclusion
"""

# 1. Installation
# Ensure that Featuretools is installed
# You can install it using pip:
# pip install featuretools

# 2. Loading Required Libraries and Data
import featuretools as ft
import pandas as pd
import numpy as np
from datetime import datetime

# Sample Data
# We'll create a more complex mock dataset to demonstrate the functionality
customers_data = {
    "customer_id": [1, 2, 3, 4, 5],
    "join_date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]),
    "age": [34, 23, 45, 25, 32],
    "gender": ["M", "F", "F", "M", "F"],
    "region": ["North", "South", "East", "West", "North"]
}

transactions_data = {
    "transaction_id": range(1, 11),
    "customer_id": [1, 2, 2, 3, 4, 4, 4, 5, 5, 5],
    "transaction_date": pd.to_datetime([
        "2023-01-05", "2023-01-06", "2023-01-07", "2023-01-08", "2023-01-09", 
        "2023-01-10", "2023-01-11", "2023-01-12", "2023-01-13", "2023-01-14"
    ]),
    "amount": [100, 150, 200, 130, 120, 170, 180, 110, 160, 190]
}

products_data = {
    "product_id": range(1, 6),
    "product_name": ["Product A", "Product B", "Product C", "Product D", "Product E"],
    "category": ["Electronics", "Electronics", "Grocery", "Grocery", "Clothing"]
}

transaction_details_data = {
    "transaction_id": [1, 1, 2, 3, 4, 5, 5, 6, 7, 8],
    "product_id": [1, 2, 2, 3, 4, 5, 1, 2, 3, 4],
    "quantity": [1, 2, 1, 1, 3, 1, 2, 1, 4, 1]
}

# Creating DataFrames
customers_df = pd.DataFrame(customers_data)
transactions_df = pd.DataFrame(transactions_data)
products_df = pd.DataFrame(products_data)
transaction_details_df = pd.DataFrame(transaction_details_data)

print("Customers DataFrame:")
print(customers_df)
print("\nTransactions DataFrame:")
print(transactions_df)
print("\nProducts DataFrame:")
print(products_df)
print("\nTransaction Details DataFrame:")
print(transaction_details_df)

# 3. Data Preparation
# Ensuring data types are correct and handling missing values if any
customers_df['customer_id'] = customers_df['customer_id'].astype(str)
transactions_df['customer_id'] = transactions_df['customer_id'].astype(str)
transaction_details_df['transaction_id'] = transaction_details_df['transaction_id'].astype(str)
transaction_details_df['product_id'] = transaction_details_df['product_id'].astype(str)

# 4. Creating an EntitySet
# An EntitySet is a collection of entities (dataframes) and relationships between them

es = ft.EntitySet(id="retail_data")

# Adding entities to the EntitySet
es = es.entity_from_dataframe(entity_id="customers",
                              dataframe=customers_df,
                              index="customer_id",
                              time_index="join_date")

es = es.entity_from_dataframe(entity_id="transactions",
                              dataframe=transactions_df,
                              index="transaction_id",
                              time_index="transaction_date")

es = es.entity_from_dataframe(entity_id="products",
                              dataframe=products_df,
                              index="product_id")

es = es.entity_from_dataframe(entity_id="transaction_details",
                              dataframe=transaction_details_df,
                              make_index=True,
                              index="detail_id")

print("\nEntitySet with all entities:")
print(es)

# 5. Deep Feature Synthesis with Multiple Tables
# Adding relationships between entities
relationship_customers_transactions = ft.Relationship(es["customers"]["customer_id"],
                                                      es["transactions"]["customer_id"])

relationship_transactions_details = ft.Relationship(es["transactions"]["transaction_id"],
                                                    es["transaction_details"]["transaction_id"])

relationship_products_details = ft.Relationship(es["products"]["product_id"],
                                                es["transaction_details"]["product_id"])

es = es.add_relationship(relationship_customers_transactions)
es = es.add_relationship(relationship_transactions_details)
es = es.add_relationship(relationship_products_details)

print("\nEntitySet with relationships:")
print(es)

# Running Deep Feature Synthesis
feature_matrix, feature_defs = ft.dfs(entityset=es,
                                      target_entity="customers",
                                      agg_primitives=["sum", "mean", "count"],
                                      trans_primitives=["day", "month", "year"])

print("\nFeature Matrix:")
print(feature_matrix)

# 6. Handling Relationships
# Understanding relationships and their implications in feature engineering

# 7. Advanced Aggregation and Transformation Primitives
# Using advanced primitives to generate more complex features

advanced_primitive_options = {
    "aggregation_primitives": ["sum", "mean", "count", "min", "max", "std", "skew", "mode"],
    "transform_primitives": ["day", "month", "year", "weekday", "is_weekend", "num_words", "num_characters"]
}

feature_matrix_advanced, feature_defs_advanced = ft.dfs(entityset=es,
                                                        target_entity="customers",
                                                        agg_primitives=advanced_primitive_options["aggregation_primitives"],
                                                        trans_primitives=advanced_primitive_options["transform_primitives"])

print("\nAdvanced Feature Matrix:")
print(feature_matrix_advanced)

# 8. Custom Primitives
# Creating custom aggregation and transformation primitives

from featuretools.primitives import AggregationPrimitive, TransformPrimitive
from featuretools.variable_types import Numeric

class Range(AggregationPrimitive):
    name = "range"
    input_types = [Numeric]
    return_type = Numeric

    def get_function(self):
        def range_func(x):
            return x.max() - x.min()
        return range_func

class Square(TransformPrimitive):
    name = "square"
    input_types = [Numeric]
    return_type = Numeric

    def get_function(self):
        def square_func(x):
            return x ** 2
        return square_func

custom_primitive_options = {
    "aggregation_primitives": ["sum", "mean", "count", Range],
    "transform_primitives": ["day", "month", "year", Square]
}

feature_matrix_custom, feature_defs_custom = ft.dfs(entityset=es,
                                                    target_entity="customers",
                                                    agg_primitives=custom_primitive_options["aggregation_primitives"],
                                                    trans_primitives=custom_primitive_options["transform_primitives"])

print("\nCustom Feature Matrix:")
print(feature_matrix_custom)

# 9. Handling Time Variables and Time-Based Features
# Featuretools can handle time variables effectively and generate time-based features

# Adding a new column with a timestamp
customers_df['last_purchase_date'] = pd.to_datetime([
    "2023-01-15", "2023-01-16", "2023-01-17", "2023-01-18", "2023-01-19"
])

# Updating the EntitySet with the new column
es = es.entity_from_dataframe(entity_id="customers",
                              dataframe=customers_df,
                              index="customer_id",
                              time_index="join_date",
                              secondary_time_index={"last_purchase_date": ['last_purchase_date']})

print("\nEntitySet with secondary time index:")
print(es)

# 10. Error Handling and Troubleshooting
# Discussing potential errors and their solutions

try:
    # Example: Trying to use a primitive that doesn't exist
    feature_matrix_error, feature_defs_error = ft.dfs(entityset=es,
                                                      target_entity="customers",
                                                      agg_primitives=["nonexistent_primitive"],
                                                      trans_primitives=["day", "month", "year"])
except Exception as e:
    print("\nError encountered during DFS:")
    print(e)

# Potential Error: Incorrect data types
# Solution: Ensure all data types are correct before running DFS

try:
    # Example: Incorrect data type for customer_id in transactions_df
    transactions_df['customer_id'] = transactions_df['customer_id'].astype(int)
    es = es.entity_from_dataframe(entity_id="transactions",
                                  dataframe=transactions_df,
                                  index="transaction_id",
                                  time_index="transaction_date")
except Exception as e:
    print("\nError encountered due to incorrect data type:")
    print(e)
    # Correcting the data type
    transactions_df['customer_id'] = transactions_df['customer_id'].astype(str)
    es = es.entity_from_dataframe(entity_id="transactions",
                                  dataframe=transactions_df,
                                  index="transaction_id",
                                  time_index="transaction_date")

# 11. Conclusion
# This tutorial covered the basics and some advanced features of Featuretools for automated feature engineering.
# You can further explore the library to tailor it to your specific needs.
