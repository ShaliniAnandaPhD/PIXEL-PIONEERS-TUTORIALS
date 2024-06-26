import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu
import numpy as np

# Simulate data
# For demonstration purposes, let's simulate some example data
np.random.seed(42)
X_train = np.random.rand(100, 8)  # 100 samples, 8 features
y_train = np.random.rand(100, 4)  # 100 samples, 4 targets (calories, proteins, carbs, fats)

X_test = np.random.rand(20, 8)    # 20 samples, 8 features
y_test = np.random.rand(20, 4)    # 20 samples, 4 targets

# Define the neural network architecture
def create_model(input_shape):
    return stax.serial(
        Dense(128), Relu,
        Dense(64), Relu,
        Dense(4)  # Output layer for 4 targets
    )

# Create the model
init_fun, apply_fun = create_model(X_train.shape[1])

# Initialize model parameters
key = random.PRNGKey(42)
_, init_params = init_fun(key, (-1, X_train.shape[1]))

# Define the loss function
def loss_fn(params, inputs, targets):
    predictions = apply_fun(params, inputs)
    return jnp.mean((predictions - targets) ** 2)

# Define the update function
@jit
def update(params, inputs, targets, learning_rate):
    grads = grad(loss_fn)(params, inputs, targets)
    return jax.tree_util.tree_multimap(lambda p, g: p - learning_rate * g, params, grads)

# Training settings
num_epochs = 100
batch_size = 32
learning_rate = 0.001

# Parallelize the update function across batch dimensions
batched_update = vmap(update, in_axes=(None, 0, 0, None))

# Train the model
for epoch in range(num_epochs):
    # Shuffle the training data
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    # Mini-batch training
    epoch_loss = 0.0
    num_batches = 0
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # Update parameters
        init_params = batched_update(init_params, X_batch, y_batch, learning_rate)
        batch_loss = loss_fn(init_params, X_batch, y_batch)
        epoch_loss += batch_loss
        num_batches += 1

    # Calculate average loss for the epoch
    epoch_loss /= num_batches
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

# Evaluate the model
test_predictions = apply_fun(init_params, X_test)
test_loss = jnp.mean((test_predictions - y_test) ** 2)
print(f"Test Loss: {test_loss:.4f}")

# Define the inference function
@jit
def predict(params, inputs):
    return apply_fun(params, inputs)

# Example usage
new_input = np.array([[5, 3, 2, 1, 4, 2, 0, 1]])  # Replace with actual input features
predicted_nutrition = predict(init_params, new_input)
print(f"Predicted Nutrition: {predicted_nutrition}")

# Possible Errors and Solutions:
# 1. Import Errors:
#    Error: "ModuleNotFoundError: No module named 'jax'"
#    Solution: Ensure JAX and other required libraries are properly installed. Use `pip install jax jaxlib flax optax`.

# 2. Shape Mismatch Errors:
#    Error: "ValueError: shapes (X,Y) and (Y,Z) not aligned"
#    Solution: Verify the shapes of inputs and weights in matrix multiplication. Adjust dimensions if necessary.

# 3. Gradient Issues:
#    Error: "ValueError: gradients must be arrays"
#    Solution: Ensure that the loss function returns a scalar value for proper gradient computation.

# 4. Performance Issues:
#    Solution: Use smaller batch sizes or fewer epochs if the training process is too slow. Consider using GPU for faster computation.

