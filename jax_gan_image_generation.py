# jax_gan_image_generation.py

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Define the generator network
def jax_generator(z, hidden_size):
    x = jax.nn.dense(z, hidden_size)
    x = jax.nn.relu(x)
    x = jax.nn.dense(x, 784)  # 28x28 = 784
    x = jax.nn.sigmoid(x)
    return x

# Define the discriminator network
def jax_discriminator(x, hidden_size):
    x = jax.nn.dense(x, hidden_size)
    x = jax.nn.relu(x)
    x = jax.nn.dense(x, 1)
    return x

# Define the loss functions
def jax_generator_loss(generator_params, discriminator_params, z):
    generated_images = jax_generator(z, hidden_size=128)
    logits = jax_discriminator(generated_images, hidden_size=128)
    loss = jnp.mean(jax.nn.sigmoid_cross_entropy_with_logits(logits, jnp.ones_like(logits)))
    return loss

def jax_discriminator_loss(generator_params, discriminator_params, real_images, z):
    generated_images = jax_generator(z, hidden_size=128)
    real_logits = jax_discriminator(real_images, hidden_size=128)
    fake_logits = jax_discriminator(generated_images, hidden_size=128)
    real_loss = jnp.mean(jax.nn.sigmoid_cross_entropy_with_logits(real_logits, jnp.ones_like(real_logits)))
    fake_loss = jnp.mean(jax.nn.sigmoid_cross_entropy_with_logits(fake_logits, jnp.zeros_like(fake_logits)))
    total_loss = real_loss + fake_loss
    return total_loss

# Load and preprocess the MNIST dataset
def jax_load_data():
    (x_train, _), (_, _) = mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    return x_train

# Train the GAN
def jax_train(generator_params, discriminator_params, real_images, num_epochs, batch_size, z_dim):
    for epoch in range(num_epochs):
        for i in range(0, len(real_images), batch_size):
            batch_images = real_images[i:i+batch_size]
            z = jax.random.normal(jax.random.PRNGKey(0), (batch_size, z_dim))
            
            # Update discriminator parameters
            discriminator_loss, discriminator_grads = jax.value_and_grad(jax_discriminator_loss)(generator_params, discriminator_params, batch_images, z)
            discriminator_params = jax.tree_multimap(lambda p, g: p - 0.001 * g, discriminator_params, discriminator_grads)
            
            # Update generator parameters
            generator_loss, generator_grads = jax.value_and_grad(jax_generator_loss)(generator_params, discriminator_params, z)
            generator_params = jax.tree_multimap(lambda p, g: p - 0.001 * g, generator_params, generator_grads)
        
        print(f"Epoch {epoch + 1}, Generator Loss: {generator_loss:.4f}, Discriminator Loss: {discriminator_loss:.4f}")
    
    return generator_params, discriminator_params

# Generate and visualize new images
def jax_generate_images(generator_params, num_images, z_dim):
    z = jax.random.normal(jax.random.PRNGKey(0), (num_images, z_dim))
    generated_images = jax_generator(z, hidden_size=128)
    generated_images = generated_images.reshape(-1, 28, 28)
    
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(10, 10, i + 1)
        plt.imshow(generated_images[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Initialize generator and discriminator parameters
    rng = jax.random.PRNGKey(0)
    generator_params = jax.random.normal(rng, (100, 128))
    discriminator_params = jax.random.normal(rng, (784, 128))
    
    # Load and preprocess data
    real_images = jax_load_data()
    
    # Train the GAN
    generator_params, discriminator_params = jax_train(generator_params, discriminator_params, real_images, num_epochs=10, batch_size=64, z_dim=100)
    
    # Generate and visualize new images
    jax_generate_images(generator_params, num_images=100, z_dim=100)

if __name__ == "__main__":
    main()
