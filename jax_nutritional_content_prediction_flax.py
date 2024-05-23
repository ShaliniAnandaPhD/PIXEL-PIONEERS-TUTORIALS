# jax_nutritional_content_prediction_flax.py

# Import necessary libraries
import jax
import jax.numpy as jnp
from jax import random, grad, jit
import flax.linen as nn
import numpy as np

# Simulate data
# For demonstration purposes, let's simulate some example data
np.random.seed(42)
X_train = np.random.rand(100, 8)  # 100 samples, 8 features
y_train = np.random.rand(100, 4)  # 100 samples, 4 targets (calories, proteins, carbs, fats)

X_test = np.random.rand(20, 8)    # 20 samples, 8 features
y_test = np.random.rand(20, 4)    # 20 samples, 4 targets

# Define the neural network architecture using Flax
class NutritionalModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(4)(x)  # Output layer for 4 targets
        return x

# Create an instance of the model
model = NutritionalModel()

# Define the loss function
def loss_fn(params, inputs, targets):
    predictions = model.apply(params, inputs)
    return jnp.mean((predictions - targets) ** 2)

# Define the update function
@jit
def update(params, inputs, targets, learning_rate):
    grads = grad(loss_fn)(params, inputs, targets)
    return jax.tree_util.tree_multimap(lambda p, g: p - learning_rate * g, params, grads)

# Initialize model parameters
key = random.PRNGKey(42)
input_shape = (X_train.shape[0], X_train.shape[1])  # Shape of the input data
params = model.init(key, jnp.ones(input_shape))['params']

# Training settings
num_epochs = 100
batch_size = 32
learning_rate = 0.001

# Train the model
for epoch in range(num_epochs):
    # Shuffle the training data
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    # Mini-batch training
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # Update parameters
        params = update(params, X_batch, y_batch, learning_rate)
    
    # Calculate loss for monitoring
    train_loss = loss_fn(params, X_train, y_train)
    print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}")

# Evaluate the model
test_predictions = model.apply(params, X_test)
test_loss = jnp.mean((test_predictions - y_test) ** 2)
print(f"Test Loss: {test_loss:.4f}")

# Define the inference function
@jit
def predict(params, inputs):
    return model.apply(params, inputs)

# Example usage
new_input = np.array([[5, 3, 2, 1, 4, 2, 0, 1]])  # Replace with actual input features
predicted_nutrition = predict(params, new_input)
print(f"Predicted Nutrition: {predicted_nutrition}")
