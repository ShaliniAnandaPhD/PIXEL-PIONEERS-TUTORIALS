# File name: jax_vae_face_generation.py

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Simulate a dataset of face images
def generate_face_dataset(num_samples):
    faces = np.random.rand(num_samples, 64, 64, 3)  # Random face images
    return faces

# Define the encoder network
def encoder(inputs):
    x = jnp.reshape(inputs, (inputs.shape[0], -1))  # Flatten the input image
    x = jax.nn.relu(jax.nn.dense(x, 512))  # Hidden layer
    mean = jax.nn.dense(x, 128)  # Output layer for mean
    log_var = jax.nn.dense(x, 128)  # Output layer for log-variance
    return mean, log_var

# Define the decoder network
def decoder(latent):
    x = jax.nn.relu(jax.nn.dense(latent, 512))  # Hidden layer
    x = jax.nn.sigmoid(jax.nn.dense(x, 64*64*3))  # Output layer
    output = jnp.reshape(x, (x.shape[0], 64, 64, 3))  # Reshape to image
    return output

# Define the VAE model
def vae(inputs):
    mean, log_var = encoder(inputs)  # Encode the input image
    latent = mean + jnp.exp(0.5 * log_var) * jax.random.normal(jax.random.PRNGKey(0), mean.shape)  # Sample latent variables
    output = decoder(latent)  # Decode the latent variables
    return output, mean, log_var

# Define the loss function
def vae_loss(inputs):
    output, mean, log_var = vae(inputs)
    reconstruction_loss = jnp.mean(jnp.square(output - inputs))  # Reconstruction loss
    kl_loss = -0.5 * jnp.mean(1 + log_var - jnp.square(mean) - jnp.exp(log_var))  # KL divergence loss
    total_loss = reconstruction_loss + kl_loss  # Total VAE loss
    return total_loss

# Train the VAE model
@jax.jit
def train_step(optimizer, inputs):
    loss, grads = jax.value_and_grad(vae_loss)(inputs)
    optimizer = optimizer.apply_gradients(grads)
    return optimizer, loss

# Generate new faces using the trained VAE model
@jax.jit
def generate_faces(latent):
    generated_faces = decoder(latent)
    return generated_faces

# Set hyperparameters
num_epochs = 100
batch_size = 32
learning_rate = 0.001

# Generate simulated face dataset
faces = generate_face_dataset(num_samples=1000)

# Initialize the optimizer
optimizer = jax.optim.Adam(learning_rate=learning_rate)

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0
    for i in range(0, len(faces), batch_size):
        batch_faces = faces[i:i+batch_size]
        optimizer, loss = train_step(optimizer, batch_faces)
        epoch_loss += loss
    epoch_loss /= (len(faces) // batch_size)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

# Generate new faces
num_generated_faces = 5
latent_samples = jax.random.normal(jax.random.PRNGKey(0), (num_generated_faces, 128))
generated_faces = generate_faces(latent_samples)

# Display the generated faces
fig, axes = plt.subplots(1, num_generated_faces, figsize=(12, 3))
for i in range(num_generated_faces):
    axes[i].imshow(generated_faces[i])
    axes[i].axis('off')
plt.tight_layout()
plt.show()

# Possible Errors and Solutions:

# ValueError: operands could not be broadcast together with shapes (x, y) (a, b)
# Solution: Ensure that the shapes of the predictions and targets match exactly when calculating the loss.

# ImportError: No module named 'jax'
# Solution: Ensure JAX is installed using `pip install jax`.

# TypeError: only integer scalar arrays can be converted to a scalar index
# Solution: Ensure that array indexing and slicing operations are correctly implemented.

# RuntimeError: Invalid argument: Non-scalable parameters
# Solution: Ensure all operations in the model are scalable and support JAX's JIT compilation.
