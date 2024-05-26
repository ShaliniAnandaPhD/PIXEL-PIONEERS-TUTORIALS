# File name: pytorch_dietary_recommendation_system_nutrition.py
# File library: PyTorch, Pandas, NumPy
# Use case: Nutrition - Dietary Recommendation System

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Simulate nutritional data
np.random.seed(42)
num_samples = 1000
age = np.random.randint(18, 80, size=num_samples)
gender = np.random.choice(['Male', 'Female'], size=num_samples)
height = np.random.normal(loc=170, scale=10, size=num_samples)
weight = np.random.normal(loc=70, scale=15, size=num_samples)
activity_level = np.random.choice(['Sedentary', 'Moderately Active', 'Active'], size=num_samples)
recommended_diet = np.random.choice(['Balanced', 'Low Carb', 'High Protein', 'Low Fat'], size=num_samples)

# Create a DataFrame with the simulated data
data = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'Height': height,
    'Weight': weight,
    'Activity Level': activity_level,
    'Recommended Diet': recommended_diet
})

# Convert categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data)

# Split the data into features and target
X = data.drop('Recommended Diet', axis=1)
y = data['Recommended Diet']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values)
y_test = torch.tensor(y_test.values)

# Define the neural network architecture
class DietRecommenderNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DietRecommenderNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Set hyperparameters
input_size = X_train.shape[1]
hidden_size = 64
output_size = len(data['Recommended Diet'].unique())
learning_rate = 0.01
num_epochs = 100

# Create an instance of the model
model = DietRecommenderNet(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f'Accuracy: {accuracy:.4f}')

# Example usage: recommend a diet for new data
new_data = torch.tensor([[35, 0, 1, 165, 60, 0, 0, 1]], dtype=torch.float32)
with torch.no_grad():
    outputs = model(new_data)
    _, predicted = torch.max(outputs.data, 1)
    recommended_diet = data['Recommended Diet'].unique()[predicted.item()]
    print(f'Recommended Diet: {recommended_diet}')
