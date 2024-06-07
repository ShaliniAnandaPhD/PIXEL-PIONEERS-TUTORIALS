# featuretools_time_series.py

"""
Tutorial: Handling Time Series Data using Featuretools

This tutorial covers:
1. Installation
2. Loading Required Libraries and Data
3. Data Preparation
4. Creating an EntitySet for Time Series Data
5. Deep Feature Synthesis with Time Series Data
6. Handling Temporal Relationships
7. Advanced Time-Based Aggregation and Transformation Primitives
8. Custom Time-Based Primitives
9. Error Handling and Troubleshooting
10. Conclusion
"""

# 1. Installation
# Ensure that Featuretools is installed
# You can install it using pip:
# pip install featuretools

# 2. Loading Required Libraries and Data
import featuretools as ft
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Sample Data
# We'll create a mock time series dataset to demonstrate the functionality
def generate_time_series_data(start_date, num_days, num_customers):
    date_range = pd.date_range(start_date, periods=num_days, freq='D')
    data = {
        "date": np.tile(date_range, num_customers),
        "customer_id": np.repeat(range(1, num_customers + 1), num_days),
        "value": np.random.randint(1, 100, num_days * num_customers)
    }
    return pd.DataFrame(data)

customers_data = {
    "customer_id": [1, 2, 3, 4, 5],
    "join_date": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04", "2022-01-05"]),
    "region": ["North", "South", "East", "West", "North"]
}

time_series_data = generate_time_series_data(start_date="2023-01-01", num_days=30, num_customers=5)

# Creating DataFrames
customers_df = pd.DataFrame(customers_data)
time_series_df = pd.DataFrame(time_series_data)

print("Customers DataFrame:")
print(customers_df)
print("\nTime Series DataFrame:")
print(time_series_df.head())

# 3. Data Preparation
# Ensuring data types are correct and handling missing values if any
customers_df['customer_id'] = customers_df['customer_id'].astype(str)
time_series_df['customer_id'] = time_series_df['customer_id'].astype(str)
time_series_df['date'] = pd.to_datetime(time_series_df['date'])

# 4. Creating an EntitySet for Time Series Data
# An EntitySet is a collection of entities (dataframes) and relationships between them

es = ft.EntitySet(id="time_series_data")

# Adding entities to the EntitySet
es = es.entity_from_dataframe(entity_id="customers",
                              dataframe=customers_df,
                              index="customer_id",
                              time_index="join_date")

es = es.entity_from_dataframe(entity_id="time_series",
                              dataframe=time_series_df,
                              make_index=True,
                              index="id",
                              time_index="date")

print("\nEntitySet with all entities:")
print(es)

# 5. Deep Feature Synthesis with Time Series Data
# Adding relationships between entities
relationship_customers_time_series = ft.Relationship(es["customers"]["customer_id"],
                                                     es["time_series"]["customer_id"])

es = es.add_relationship(relationship_customers_time_series)

print("\nEntitySet with relationships:")
print(es)

# Running Deep Feature Synthesis
feature_matrix, feature_defs = ft.dfs(entityset=es,
                                      target_entity="customers",
                                      agg_primitives=["sum", "mean", "count"],
                                      trans_primitives=["day", "month", "year"])

print("\nFeature Matrix:")
print(feature_matrix)

# 6. Handling Temporal Relationships
# Understanding temporal relationships and their implications in feature engineering

# 7. Advanced Time-Based Aggregation and Transformation Primitives
# Using advanced primitives to generate more complex time-based features

advanced_primitive_options = {
    "aggregation_primitives": ["sum", "mean", "count", "min", "max", "std", "skew", "mode", "trend"],
    "transform_primitives": ["day", "month", "year", "weekday", "is_weekend"]
}

feature_matrix_advanced, feature_defs_advanced = ft.dfs(entityset=es,
                                                        target_entity="customers",
                                                        agg_primitives=advanced_primitive_options["aggregation_primitives"],
                                                        trans_primitives=advanced_primitive_options["transform_primitives"])

print("\nAdvanced Feature Matrix:")
print(feature_matrix_advanced)

# 8. Custom Time-Based Primitives
# Creating custom time-based aggregation and transformation primitives

from featuretools.primitives import AggregationPrimitive, TransformPrimitive
from featuretools.variable_types import Numeric, Datetime

class TimeSinceLastPurchase(AggregationPrimitive):
    name = "time_since_last_purchase"
    input_types = [Datetime]
    return_type = Numeric

    def get_function(self):
        def time_since_last_purchase(dates):
            return (datetime.now() - dates.max()).days
        return time_since_last_purchase

class TimeOfDay(TransformPrimitive):
    name = "time_of_day"
    input_types = [Datetime]
    return_type = Numeric

    def get_function(self):
        def time_of_day(dates):
            return dates.dt.hour + dates.dt.minute / 60.0
        return time_of_day

custom_primitive_options = {
    "aggregation_primitives": ["sum", "mean", "count", TimeSinceLastPurchase],
    "transform_primitives": ["day", "month", "year", TimeOfDay]
}

feature_matrix_custom, feature_defs_custom = ft.dfs(entityset=es,
                                                    target_entity="customers",
                                                    agg_primitives=custom_primitive_options["aggregation_primitives"],
                                                    trans_primitives=custom_primitive_options["transform_primitives"])

print("\nCustom Feature Matrix:")
print(feature_matrix_custom)

# 9. Error Handling and Troubleshooting
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
    # Example: Incorrect data type for customer_id in time_series_df
    time_series_df['customer_id'] = time_series_df['customer_id'].astype(int)
    es = es.entity_from_dataframe(entity_id="time_series",
                                  dataframe=time_series_df,
                                  make_index=True,
                                  index="id",
                                  time_index="date")
except Exception as e:
    print("\nError encountered due to incorrect data type:")
    print(e)
    # Correcting the data type
    time_series_df['customer_id'] = time_series_df['customer_id'].astype(str)
    es = es.entity_from_dataframe(entity_id="time_series",
                                  dataframe=time_series_df,
                                  make_index=True,
                                  index="id",
                                  time_index="date")

# 10. Conclusion
# Featuretools simplifies the process of feature engineering for time series data by automating the creation of meaningful features.
# This tutorial covered the basics of using Featuretools with time series data, including creating an EntitySet, handling temporal relationships,
# using advanced primitives, creating custom primitives, and error handling.

print("Time series feature engineering with Featuretools complete.")
