import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define the LSTM model
def jax_lstm_model(inputs, hidden_size, num_layers):
    # Define a simple LSTM model with two layers
    lstm_cell = jax.experimental.stax.serial(
        jax.experimental.stax.LSTM(hidden_size),
        jax.experimental.stax.LSTM(hidden_size)
    )
    outputs, _ = jax.lax.scan(lstm_cell, inputs)
    outputs = jax.nn.dense(outputs[-1], 1)  # Predict the next value in the sequence
    return outputs

# Define the loss function
def jax_loss_fn(params, inputs, targets):
    predictions = jax_lstm_model(inputs, hidden_size=64, num_layers=2)
    loss = jnp.mean(jax.lax.square(predictions - targets))  # Mean squared error loss
    return loss

# Prepare the data
def prepare_data(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return jnp.array(X), jnp.array(y)

# Train the model
def jax_train(params, optimizer, X_train, y_train, num_epochs, batch_size):
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            loss_value, grads = jax.value_and_grad(jax_loss_fn)(params, batch_X, batch_y)
            params = optimizer.update(grads, params)
            epoch_loss += loss_value
        epoch_loss /= (len(X_train) // batch_size)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    return params

# Make predictions
def jax_predict(params, X_test):
    predictions = jax_lstm_model(X_test, hidden_size=64, num_layers=2)
    return predictions

# Example usage
data = pd.read_csv('stock_prices.csv')  # Load stock price data
prices = data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)  # Normalize the data

lookback = 30  # Number of previous days to consider for predicting the next day
X, y = prepare_data(scaled_prices, lookback)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

rng = jax.random.PRNGKey(0)
params = jax.random.normal(rng, (lookback, 1))

optimizer = jax.optim.Adam(learning_rate=0.001)

params = jax_train(params, optimizer, X_train, y_train, num_epochs=10, batch_size=32)

predictions = jax_predict(params, X_test)
predictions = scaler.inverse_transform(predictions)

print("Actual prices:", scaler.inverse_transform(y_test.reshape(-1, 1)))
print("Predicted prices:", predictions)

# Possible Errors and Solutions:

# 1. Import Errors:
#    Error: "ModuleNotFoundError: No module named 'jax'"
#    Solution: Ensure JAX and other required libraries are properly installed. Use `pip install jax jaxlib`.

# 2. Data Preparation Errors:
#    Error: "IndexError: index out of bounds"
#    Solution: Ensure that the lookback period is appropriate for the dataset size.

# 3. Shape Mismatch Errors:
#    Error: "ValueError: shapes (X,Y) and (Y,Z) not aligned"
#    Solution: Verify the shapes of inputs and weights in matrix multiplication. Adjust dimensions if necessary.

# 4. Gradient Issues:
#    Error: "ValueError: gradients must be arrays"
#    Solution: Ensure that the loss function returns a scalar value for proper gradient computation.

# 5. Performance Issues:
#    Solution: Use smaller batch sizes or fewer epochs if the training process is too slow. Consider using GPU for faster computation.

