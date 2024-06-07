# dask_featuretools_parallel.py

"""
Tutorial: Parallelizing Feature Engineering Computations with Dask and Featuretools for Space Exploration Data

This tutorial covers:
1. Installation
2. Loading Required Libraries and Data
3. Data Preparation
4. Setting Up Dask for Parallel Computation
5. Creating an EntitySet with Featuretools
6. Deep Feature Synthesis with Dask and Featuretools
7. Advanced Aggregation and Transformation Primitives
8. Custom Primitives with Parallelization
9. Error Handling and Troubleshooting
10. Conclusion
"""

# 1. Installation
# Ensure that Dask and Featuretools are installed
# You can install them using pip:
# pip install dask[complete] featuretools

# 2. Loading Required Libraries and Data
import pandas as pd
import featuretools as ft
import dask.dataframe as dd
from dask.distributed import Client

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

# Converting the DataFrame to a Dask DataFrame for parallel processing
ddf = dd.from_pandas(df, npartitions=4)

# 4. Setting Up Dask for Parallel Computation
# Setting up a Dask client to manage workers
client = Client()
print(client)

# 5. Creating an EntitySet with Featuretools
# Creating an EntitySet with Featuretools for the space exploration data
es = ft.EntitySet(id="space_exploration")

# Adding the Dask DataFrame to the EntitySet
es = es.entity_from_dataframe(entity_id="missions",
                              dataframe=ddf,
                              index="mission",
                              time_index="year")

print("\nEntitySet:")
print(es)

# 6. Deep Feature Synthesis with Dask and Featuretools
# Running Deep Feature Synthesis with Dask for parallel computation

# Defining aggregation and transformation primitives
agg_primitives = ["sum", "mean", "count"]
trans_primitives = ["year", "month", "day"]

# Running DFS
feature_matrix, feature_defs = ft.dfs(entityset=es,
                                      target_entity="missions",
                                      agg_primitives=agg_primitives,
                                      trans_primitives=trans_primitives,
                                      dask_kwargs={"cluster": client})

print("\nFeature Matrix:")
print(feature_matrix.compute())

# 7. Advanced Aggregation and Transformation Primitives
# Using advanced primitives to generate more complex features

advanced_agg_primitives = ["sum", "mean", "count", "min", "max", "std", "skew", "mode"]
advanced_trans_primitives = ["year", "month", "day", "weekday", "is_weekend"]

feature_matrix_advanced, feature_defs_advanced = ft.dfs(entityset=es,
                                                        target_entity="missions",
                                                        agg_primitives=advanced_agg_primitives,
                                                        trans_primitives=advanced_trans_primitives,
                                                        dask_kwargs={"cluster": client})

print("\nAdvanced Feature Matrix:")
print(feature_matrix_advanced.compute())

# 8. Custom Primitives with Parallelization
# Creating custom primitives and using them with Dask

from featuretools.primitives import AggregationPrimitive, TransformPrimitive
from featuretools.variable_types import Numeric, Datetime

class TimeSinceFirstLaunch(AggregationPrimitive):
    name = "time_since_first_launch"
    input_types = [Datetime]
    return_type = Numeric

    def get_function(self):
        def time_since_first_launch(dates):
            return (datetime.now() - dates.min()).days
        return time_since_first_launch

class HourOfDay(TransformPrimitive):
    name = "hour_of_day"
    input_types = [Datetime]
    return_type = Numeric

    def get_function(self):
        def hour_of_day(dates):
            return dates.dt.hour
        return hour_of_day

custom_agg_primitives = ["sum", "mean", "count", TimeSinceFirstLaunch]
custom_trans_primitives = ["year", "month", "day", HourOfDay]

feature_matrix_custom, feature_defs_custom = ft.dfs(entityset=es,
                                                    target_entity="missions",
                                                    agg_primitives=custom_agg_primitives,
                                                    trans_primitives=custom_trans_primitives,
                                                    dask_kwargs={"cluster": client})

print("\nCustom Feature Matrix:")
print(feature_matrix_custom.compute())

# 9. Error Handling and Troubleshooting
# Discussing potential errors and their solutions

# Potential Error: Dask client connection issues
# Solution: Ensure the Dask client is properly initialized and connected

try:
    # Example: Invalid client setup
    client_error = Client("invalid_address")
    feature_matrix_error, feature_defs_error = ft.dfs(entityset=es,
                                                      target_entity="missions",
                                                      agg_primitives=agg_primitives,
                                                      trans_primitives=trans_primitives,
                                                      dask_kwargs={"cluster": client_error})
except Exception as e:
    print("\nError encountered due to invalid client setup:")
    print(e)

# Potential Error: Incompatible data types
# Solution: Ensure all data types are compatible with Dask and Featuretools

try:
    # Example: Incorrect data type
    df_error = df.copy()
    df_error['year'] = df_error['year'].astype(str)
    ddf_error = dd.from_pandas(df_error, npartitions=4)
    es_error = ft.EntitySet(id="space_exploration")
    es_error = es_error.entity_from_dataframe(entity_id="missions",
                                              dataframe=ddf_error,
                                              index="mission",
                                              time_index="year")
    feature_matrix_error, feature_defs_error = ft.dfs(entityset=es_error,
                                                      target_entity="missions",
                                                      agg_primitives=agg_primitives,
                                                      trans_primitives=trans_primitives,
                                                      dask_kwargs={"cluster": client})
except Exception as e:
    print("\nError encountered due to incompatible data type:")
    print(e)

# 10. Conclusion
# Dask and Featuretools simplify the process of parallelizing feature engineering computations for large datasets.
# This tutorial covered the basics of using Dask with Featuretools for parallel computation, creating an EntitySet, running DFS,
# using advanced primitives, creating custom primitives, and handling potential errors.

print("Parallelized feature engineering with Dask and Featuretools complete.")
