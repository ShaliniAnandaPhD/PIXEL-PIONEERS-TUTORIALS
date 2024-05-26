# Python Growing Food on Mars: Analyzing Plant Growth Data and Optimizing Yield

# Welcome to the ninth class in the "Growing Food on Mars" series!
# In this class, we will learn how to use Python to analyze plant growth data and optimize yield on Mars.

# First, let's import the necessary libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Let's create a sample dataset of plant growth data:
data = {
   "Plant_ID": [1, 2, 3, 4, 5],
   "Height_cm": [10, 15, 12, 8, 20],
   "Leaves_Count": [4, 6, 5, 3, 8],
   "Fruit_Count": [0, 2, 1, 0, 4]
}

# Now, let's create a pandas DataFrame from the data:
df = pd.DataFrame(data)
print("Plant Growth Data:")
print(df)

# Let's calculate some basic statistics:
print("\nBasic Statistics:")
print(df.describe())

# Now, let's visualize the data using a scatter plot:
plt.figure(figsize=(8, 6))
plt.scatter(df["Height_cm"], df["Fruit_Count"])
plt.xlabel("Plant Height (cm)")
plt.ylabel("Fruit Count")
plt.title("Plant Height vs. Fruit Count")
plt.show()

# Let's create a function to predict the yield based on plant height:
def predict_yield(height):
   # For simplicity, we'll assume a linear relationship between height and fruit count
   slope = 0.2
   intercept = -1
   return slope * height + intercept

# Let's use the function to predict the yield for a plant with a height of 25 cm:
height = 25
predicted_yield = predict_yield(height)
print(f"\nPredicted yield for a plant with a height of {height} cm: {predicted_yield:.2f} fruits")

# Now, let's optimize the yield by finding the optimal plant height:
def optimize_yield(heights):
   max_yield = 0
   optimal_height = 0
   for height in heights:
       predicted_yield = predict_yield(height)
       if predicted_yield > max_yield:
           max_yield = predicted_yield
           optimal_height = height
   return optimal_height, max_yield

heights = np.arange(10, 31)  # Generate heights from 10 cm to 30 cm
optimal_height, max_yield = optimize_yield(heights)
print(f"\nOptimal plant height for maximum yield: {optimal_height} cm")
print(f"Maximum predicted yield: {max_yield:.2f} fruits")

# Follow along and rewrite this code in your own Python file.
# Feel free to experiment with different datasets and prediction models to further optimize yield.

# In the next class, we will explore how to implement a Python-based early warning system for plant diseases in a Martian greenhouse.
# Get ready to dive deeper into the world of proactive plant health management on Mars!
