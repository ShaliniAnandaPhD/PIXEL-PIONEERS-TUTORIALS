# File name: pytorch_wind_turbine_fault_detection_renewable_energy.py
# File library: PyTorch, NumPy, Matplotlib
# Use case: Renewable Energy - Wind Turbine Fault Detection

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate simulated wind turbine data
num_samples = 1000
normal_data = np.random.normal(loc=0, scale=1, size=(num_samples, 4))
faulty_data = np.random.normal(loc=2, scale=1, size=(num_samples, 4))

# Combine normal and faulty data
data = np.concatenate((normal_data, faulty_data), axis=0)
labels = np.concatenate((np.zeros(num_samples), np.ones(num_samples)), axis=0)

# Convert data to PyTorch tensors
data = torch.tensor(data, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

# Define the neural network architecture
class FaultDetectionNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FaultDetectionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Set hyperparameters
input_size = 4
hidden_size = 16
output_size = 2
learning_rate = 0.01
num_epochs = 100

# Create an instance of the model
model = FaultDetectionNet(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(data)
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    outputs = model(data)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == labels).sum().item() / len(labels)
    print(f'Accuracy: {accuracy:.4f}')

# Visualize the decision boundary
xx, yy = np.meshgrid(np.linspace(-4, 6, 100), np.linspace(-4, 6, 100))
grid_data = torch.tensor(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx).ravel(), np.zeros_like(xx).ravel()], dtype=torch.float32)
with torch.no_grad():
    outputs = model(grid_data)
    _, predicted = torch.max(outputs.data, 1)
    z = predicted.reshape(xx.shape)

plt.contourf(xx, yy, z, cmap=plt.cm.RdYlBu)
plt.scatter(normal_data[:, 0], normal_data[:, 1], c='blue', label='Normal')
plt.scatter(faulty_data[:, 0], faulty_data[:, 1], c='red', label='Faulty')
plt.xlabel('Sensor 1')
plt.ylabel('Sensor 2')
plt.title('Wind Turbine Fault Detection')
plt.legend()
plt.show()
