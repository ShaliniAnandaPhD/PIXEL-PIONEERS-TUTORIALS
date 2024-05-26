# File name: sklearn_nutritional_content_analysis_nutrition.py
# File library: Scikit-learn, Pandas, Seaborn
# Use case: Nutrition - Nutritional Content Analysis

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Simulate nutritional content data
np.random.seed(42)
num_samples = 100
foods = ['Apple', 'Banana', 'Carrot', 'Spinach', 'Chicken', 'Beef', 'Salmon', 'Bread', 'Rice', 'Pasta']
calories = np.random.randint(50, 500, size=num_samples)
protein = np.random.randint(0, 30, size=num_samples)
fat = np.random.randint(0, 30, size=num_samples)
carbs = np.random.randint(0, 100, size=num_samples)

# Create a DataFrame with the simulated data
data = pd.DataFrame({
    'Food': np.random.choice(foods, size=num_samples),
    'Calories': calories,
    'Protein': protein,
    'Fat': fat,
    'Carbs': carbs
})

# Preprocess the data
scaler = StandardScaler()
X = data.drop('Food', axis=1)
X_scaled = scaler.fit_transform(X)

# Perform K-means clustering
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_scaled)

# Add cluster labels to the DataFrame
data['Cluster'] = kmeans.labels_

# Visualize the clustering results
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='Calories', y='Protein', hue='Cluster', style='Cluster', palette='viridis')
plt.title('Nutritional Content Clustering')
plt.xlabel('Calories')
plt.ylabel('Protein')
plt.show()

# Analyze nutritional content by cluster
for cluster in range(n_clusters):
    cluster_data = data[data['Cluster'] == cluster]
    print(f"Cluster {cluster}:")
    print(cluster_data.describe())
    print()

# Predict the cluster for new food items
new_foods = pd.DataFrame({
    'Food': ['Yogurt', 'Oatmeal', 'Egg'],
    'Calories': [150, 200, 70],
    'Protein': [10, 5, 6],
    'Fat': [2, 3, 5],
    'Carbs': [20, 30, 1]
})

new_foods_scaled = scaler.transform(new_foods.drop('Food', axis=1))
new_foods['Cluster'] = kmeans.predict(new_foods_scaled)

print("Predicted Clusters for New Foods:")
print(new_foods)
