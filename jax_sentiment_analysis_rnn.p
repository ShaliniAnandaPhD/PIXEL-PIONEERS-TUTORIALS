# File name: jax_sentiment_analysis_rnn.py
# File library: JAX, NumPy, TensorFlow
# Use case: Sentiment Analysis with RNN

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Define the RNN model architecture
def jax_rnn_model(inputs, hidden_size):
    # Embedding layer
    embedding = jax.nn.embedding(inputs, embedding_dim=100, num_embeddings=10000)
    
    # LSTM layer
    lstm_cell = jax.experimental.stax.LSTM(hidden_size)
    outputs, _ = jax.lax.scan(lstm_cell, embedding, length=inputs.shape[1])
    
    # Output layer
    outputs = outputs[-1]  # Take the last output of the LSTM
    outputs = jax.nn.dense(outputs, features=1)
    outputs = jnp.squeeze(outputs)
    return outputs

# Define the loss function
def jax_loss_fn(params, inputs, labels):
    logits = jax_rnn_model(inputs, hidden_size=128)
    loss = jnp.mean(jax.nn.sigmoid_cross_entropy_with_logits(logits, labels))
    return loss

# Define the accuracy metric
def jax_accuracy(params, inputs, labels):
    logits = jax_rnn_model(inputs, hidden_size=128)
    predictions = jnp.round(jax.nn.sigmoid(logits))
    return jnp.mean(predictions == labels)

# Load and preprocess the IMDB dataset
def jax_load_data():
    max_features = 10000
    max_len = 500
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    return (x_train, y_train), (x_test, y_test)

# Train the model
def jax_train(params, optimizer, x_train, y_train, num_epochs):
    for epoch in range(num_epochs):
        loss_value, grads = jax.value_and_grad(jax_loss_fn)(params, x_train, y_train)
        params = optimizer.update(grads, params)
        print(f"Epoch {epoch + 1}, Loss: {loss_value:.4f}")
    return params

# Evaluate the model
def jax_evaluate(params, x_test, y_test):
    accuracy = jax_accuracy(params, x_test, y_test)
    return accuracy

# Main function
def main():
    # Initialize model parameters
    rng = jax.random.PRNGKey(0)
    params = jax.random.normal(rng, (10000, 100))

    # Initialize optimizer
    optimizer = jax.optim.Adam(learning_rate=0.001)

    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = jax_load_data()

    # Train the model
    params = jax_train(params, optimizer, x_train, y_train, num_epochs=5)

    # Evaluate the model
    accuracy = jax_evaluate(params, x_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()

# Possible errors and solutions:
# 1. TensorFlow dataset loading issues:
#    Error: "ModuleNotFoundError: No module named 'tensorflow_datasets'"
#    Solution: Ensure TensorFlow is properly installed. Use `pip install tensorflow` to install it.
#
# 2. Shape mismatch errors during training:
#    Error: "ValueError: operands could not be broadcast together with shapes..."
#    Solution: Check the shapes of the inputs and labels to ensure they are compatible. Use appropriate reshaping or padding if necessary.
#
# 3. Slow training or convergence issues:
#    Solution: Experiment with different learning rates, batch sizes, or network architectures. Use a smaller model or fewer parameters if the training is too slow.
