# jax_autoencoder_anomaly_detection.py

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define the autoencoder model
def jax_autoencoder(inputs, hidden_size):
    encoder = jax.nn.dense(inputs, hidden_size, activation=jax.nn.relu)
    decoder = jax.nn.dense(encoder, inputs.shape[-1])
    return decoder

# Define the loss function
def jax_loss_fn(params, inputs):
    reconstructed = jax_autoencoder(inputs, hidden_size=32)
    loss = jnp.mean(jax.lax.square(reconstructed - inputs))
    return loss

# Train the autoencoder
def jax_train(params, optimizer, X_train, num_epochs, batch_size):
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            loss_value, grads = jax.value_and_grad(jax_loss_fn)(params, batch_X)
            params = optimizer.update(grads, params)
            epoch_loss += loss_value
        epoch_loss /= (len(X_train) // batch_size)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    return params

# Detect anomalies
def jax_detect_anomalies(params, X, threshold):
    reconstructed = jax_autoencoder(X, hidden_size=32)
    reconstruction_error = jnp.mean(jax.lax.square(reconstructed - X), axis=-1)
    anomalies = jnp.where(reconstruction_error > threshold, 1, 0)
    return anomalies

# Example usage
X, y = make_classification(n_samples=1000, n_classes=2, n_features=10, n_informative=8, n_redundant=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rng = jax.random.PRNGKey(0)
params = jax.random.normal(rng, (X_train.shape[-1], 32))

optimizer = jax.optim.Adam(learning_rate=0.001)

params = jax_train(params, optimizer, X_train, num_epochs=10, batch_size=32)

threshold = 0.1
anomalies = jax_detect_anomalies(params, X_test, threshold)

accuracy = accuracy_score(y_test, anomalies)
print("Accuracy:", accuracy)
