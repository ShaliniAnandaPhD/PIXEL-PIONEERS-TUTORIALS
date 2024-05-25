# File name: jax_vae_face_generation.py

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Simulated dataset of face images
def generate_face_dataset(num_samples):
    # Generate random face images (simulated)
    faces = np.random.rand(num_samples, 64, 64, 3)
    return faces

# Define the encoder network
def encoder(inputs):
    # Flatten the input image
    x = jnp.reshape(inputs, (inputs.shape[0], -1))
    # Encoder hidden layer
    x = jax.nn.relu(jax.nn.dense(x, 512))
    # Encoder output layers for mean and log-variance
    mean = jax.nn.dense(x, 128)
    log_var = jax.nn.dense(x, 128)
    return mean, log_var

# Define the decoder network
def decoder(latent):
    # Decoder hidden layer
    x = jax.nn.relu(jax.nn.dense(latent, 512))
    # Decoder output layer
    x = jax.nn.sigmoid(jax.nn.dense(x, 64*64*3))
    # Reshape the output to the original image shape
    output = jnp.reshape(x, (x.shape[0], 64, 64, 3))
    return output

# Define the VAE model
def vae(inputs):
    # Encode the input image
    mean, log_var = encoder(inputs)
    # Sample latent variables from the encoded distribution
    latent = mean + jnp.exp(0.5 * log_var) * jax.random.normal(jax.random.PRNGKey(0), mean.shape)
    # Decode the latent variables to generate the output image
    output = decoder(latent)
    return output, mean, log_var

# Define the loss function
def vae_loss(inputs):
    output, mean, log_var = vae(inputs)
    # Reconstruction loss
    reconstruction_loss = jnp.mean(jnp.square(output - inputs))
    # KL divergence loss
    kl_loss = -0.5 * jnp.mean(1 + log_var - jnp.square(mean) - jnp.exp(log_var))
    # Total VAE loss
    total_loss = reconstruction_loss + kl_loss
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
    print(f"Epoch {epoch+1}, Loss: {epoch_loss}")

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
