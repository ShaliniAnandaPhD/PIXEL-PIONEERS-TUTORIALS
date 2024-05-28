
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Define the linear regression model
def linear_regression(params, features):
    weights, bias = params
    predictions = jnp.dot(features, weights) + bias
    return predictions

# Define the loss function (mean squared error)
def mse_loss(params, features, targets):
    predictions = linear_regression(params, features)
    loss = jnp.mean(jnp.square(predictions - targets))
    return loss

# Define the training step
@jax.jit
def train_step(params, features, targets, learning_rate):
    loss, gradients = jax.value_and_grad(mse_loss)(params, features, targets)
    weights, bias = params
    weights -= learning_rate * gradients[0]
    bias -= learning_rate * gradients[1]
    return (weights, bias), loss

# Load and preprocess the California Housing dataset
data = fetch_california_housing()
features = data.data
targets = data.target

# Split the data into training and testing sets
features_train, features_test, targets_train, targets_test = train_test_split(
    features, targets, test_size=0.2, random_state=42)

# Normalize the features
mean = np.mean(features_train, axis=0)
std = np.std(features_train, axis=0)
features_train = (features_train - mean) / std
features_test = (features_test - mean) / std

# Set hyperparameters
num_epochs = 100
learning_rate = 0.01

# Initialize model parameters
weights = jnp.zeros(features_train.shape[1])
bias = 0.0
params = (weights, bias)

# Training loop
for epoch in range(num_epochs):
    params, loss = train_step(params, features_train, targets_train, learning_rate)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Evaluate the model on the test set
predictions_test = linear_regression(params, features_test)
mse = mean_squared_error(targets_test, predictions_test)
r2 = r2_score(targets_test, predictions_test)

print(f"Test MSE: {mse:.4f}")
print(f"Test R^2: {r2:.4f}")

# Possible Errors and Solutions:

# 1. ValueError: If there are NaN values in the dataset, you might get a ValueError.
#    Solution: Ensure that your dataset does not contain NaN values by using `np.nan_to_num` or similar preprocessing steps.

# 2. Dimension Mismatch: If the dimensions of weights or features do not align, an error will occur.
#    Solution: Check the shapes of your arrays to ensure they are correct, especially after splitting the data.

# 3. Convergence Issues: If the learning rate is too high, the model may not converge and result in a high loss.
#    Solution: Reduce the learning rate and observe the change in loss over epochs.

# 4. Memory Issues: For large datasets, you might encounter memory issues.
#    Solution: Use batch processing or reduce the dataset size.
