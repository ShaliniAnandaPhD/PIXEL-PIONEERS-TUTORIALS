# File name: jax_prototypical_networks.py
# File library: JAX, NumPy, Flax, TensorFlow Datasets
# Use case: Prototypical Networks for Few-Shot Learning

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import train_state
import optax
from tensorflow_datasets import load_dataset

# Define the embedding network
class EmbeddingNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        return x

# Define the prototypical network
class PrototypicalNetwork(nn.Module):
    num_classes: int
    num_support: int
    num_query: int

    @nn.compact
    def __call__(self, support_set, query_set):
        # Compute embeddings for support and query sets
        support_embeddings = EmbeddingNetwork()(support_set)
        query_embeddings = EmbeddingNetwork()(query_set)

        # Compute prototypes for each class
        prototypes = jnp.stack([jnp.mean(support_embeddings[i:i+self.num_support], axis=0) 
                                for i in range(0, self.num_classes * self.num_support, self.num_support)])

        # Compute distances between query embeddings and prototypes
        distances = jnp.sqrt(jnp.sum(jnp.square(query_embeddings[:, jnp.newaxis, :] - prototypes), axis=-1))
        
        # Compute log probabilities
        log_probabilities = -distances
        log_probabilities = nn.log_softmax(log_probabilities)

        return log_probabilities

# Define the loss function
def loss_fn(params, support_set, query_set, query_labels):
    log_probabilities = PrototypicalNetwork(num_classes, num_support, num_query)(params, support_set, query_set)
    loss = -jnp.mean(jnp.sum(jax.nn.one_hot(query_labels, num_classes) * log_probabilities, axis=-1))
    return loss

# Define the evaluation metrics
def accuracy(params, support_set, query_set, query_labels):
    log_probabilities = PrototypicalNetwork(num_classes, num_support, num_query)(params, support_set, query_set)
    predicted_labels = jnp.argmax(log_probabilities, axis=-1)
    accuracy = jnp.mean(predicted_labels == query_labels)
    return accuracy

# Load and preprocess the Omniglot dataset
def load_omniglot_data(num_train_examples, num_test_examples):
    ds_train = load_dataset("omniglot", split="train[:90%]")
    ds_test = load_dataset("omniglot", split="test")

    def preprocess_data(example):
        image = example["image"] / 255.0
        label = example["alphabet"]
        return image, label

    ds_train = ds_train.map(preprocess_data).shuffle(1024).batch(num_train_examples)
    ds_test = ds_test.map(preprocess_data).batch(num_test_examples)

    return ds_train, ds_test

# Set hyperparameters
num_classes = 5
num_support = 5
num_query = 15
num_train_examples = num_classes * (num_support + num_query)
num_test_examples = num_classes * (num_support + num_query)
num_epochs = 10
learning_rate = 0.001

# Load and preprocess the Omniglot dataset
ds_train, ds_test = load_omniglot_data(num_train_examples, num_test_examples)

# Create the model and optimizer
model = PrototypicalNetwork(num_classes, num_support, num_query)
params = model.init(jax.random.PRNGKey(0), jnp.zeros((num_train_examples, 28, 28, 1)), jnp.zeros((num_query * num_classes, 28, 28, 1)))
tx = optax.adam(learning_rate)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for support_images, support_labels in ds_train:
        # Create the support set
        support_set = jnp.concatenate([support_images[support_labels == i][:num_support] for i in range(num_classes)], axis=0)
        
        # Create the query set
        query_set = jnp.concatenate([support_images[support_labels == i][num_support:num_support+num_query] for i in range(num_classes)], axis=0)
        query_labels = jnp.concatenate([jnp.full((num_query,), i) for i in range(num_classes)], axis=0)

        # Compute the loss and update the model parameters
        loss_value, grads = jax.value_and_grad(loss_fn)(state.params, support_set, query_set, query_labels)
        state = state.apply_gradients(grads=grads)
        epoch_loss += loss_value

    epoch_loss /= len(ds_train)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Evaluation loop
accuracies = []
for support_images, support_labels in ds_test:
    # Create the support set
    support_set = jnp.concatenate([support_images[support_labels == i][:num_support] for i in range(num_classes)], axis=0)
    
    # Create the query set
    query_set = jnp.concatenate([support_images[support_labels == i][num_support:num_support+num_query] for i in range(num_classes)], axis=0)
    query_labels = jnp.concatenate([jnp.full((num_query,), i) for i in range(num_classes)], axis=0)

    # Compute the accuracy
    accuracy_value = accuracy(state.params, support_set, query_set, query_labels)
    accuracies.append(accuracy_value)

mean_accuracy = jnp.mean(jnp.array(accuracies))
print(f"Test Accuracy: {mean_accuracy:.4f}")

# Possible errors and solutions:
# 1. Import Errors:
#    Error: "ModuleNotFoundError: No module named 'flax'"
#    Solution: Ensure Flax is properly installed. Use `pip install flax` to install it.
#
# 2. Data Loading Issues:
#    Error: "ModuleNotFoundError: No module named 'tensorflow_datasets'"
#    Solution: Ensure TensorFlow Datasets is properly installed. Use `pip install tensorflow-datasets` to install it.
#
# 3. Shape Mismatch Errors:
#    Error: "ValueError: operands could not be broadcast together with shapes..."
#    Solution: Check the shapes of the inputs and ensure they match the expected shapes for the model. Adjust the data preprocessing steps if necessary.
#
# 4. Slow Training:
#    Solution: Experiment with different learning rates, batch sizes, or number of epochs. Use a smaller model or fewer parameters if the training is too slow.

