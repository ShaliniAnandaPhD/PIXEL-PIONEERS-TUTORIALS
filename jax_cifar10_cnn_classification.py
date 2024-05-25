# jax_cifar10_cnn_classification.py

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds

# Define the CNN model architecture
def jax_cnn_model(inputs):
    # Convolutional layer 1
    x = jax.lax.conv(inputs, kernel_size=(3, 3), feature_map_shape=(32,), padding='SAME')
    x = jax.nn.relu(x)
    x = jax.lax.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')

    # Convolutional layer 2
    x = jax.lax.conv(x, kernel_size=(3, 3), feature_map_shape=(64,), padding='SAME')
    x = jax.nn.relu(x)
    x = jax.lax.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')

    # Flatten the feature maps
    x = jnp.reshape(x, (x.shape[0], -1))

    # Fully connected layer
    x = jax.nn.dense(x, features=256)
    x = jax.nn.relu(x)

    # Output layer
    x = jax.nn.dense(x, features=10)
    return x

# Define the loss function
def jax_loss_fn(params, inputs, labels):
    logits = jax_cnn_model(inputs)
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    loss = jnp.mean(jax.nn.softmax_cross_entropy_with_logits(logits, one_hot_labels))
    return loss

# Define the accuracy metric
def jax_accuracy(params, inputs, labels):
    logits = jax_cnn_model(inputs)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)

# Load and preprocess the CIFAR-10 dataset
def jax_load_data():
    ds_train, ds_test = tfds.load('cifar10', split=['train', 'test'], as_supervised=True)
    ds_train = ds_train.map(lambda x, y: (x / 255.0, y)).batch(32).prefetch(1)
    ds_test = ds_test.map(lambda x, y: (x / 255.0, y)).batch(32).prefetch(1)
    return ds_train, ds_test

# Train the model
def jax_train(params, optimizer, ds_train, num_epochs):
    for epoch in range(num_epochs):
        for batch in ds_train:
            inputs, labels = batch
            loss_value, grads = jax.value_and_grad(jax_loss_fn)(params, inputs, labels)
            params = optimizer.update(grads, params)
        print(f"Epoch {epoch + 1}, Loss: {loss_value:.4f}")
    return params

# Evaluate the model
def jax_evaluate(params, ds_test):
    accuracies = []
    for batch in ds_test:
        inputs, labels = batch
        acc = jax_accuracy(params, inputs, labels)
        accuracies.append(acc)
    return jnp.mean(jnp.array(accuracies))

# Main function
def main():
    # Initialize model parameters
    rng = jax.random.PRNGKey(0)
    params = jax.random.normal(rng, (3, 32, 32, 32))

    # Initialize optimizer
    optimizer = jax.optim.Adam(learning_rate=0.001)

    # Load and preprocess data
    ds_train, ds_test = jax_load_data()

    # Train the model
    params = jax_train(params, optimizer, ds_train, num_epochs=10)

    # Evaluate the model
    accuracy = jax_evaluate(params, ds_test)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
