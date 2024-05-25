# jax_image_inpainting.py

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import train_state
import optax
from tensorflow_datasets import load_dataset
import matplotlib.pyplot as plt

# Define the generator network
class Generator(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Conv(features=128, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Conv(features=256, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.ConvTranspose(features=128, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.ConvTranspose(features=64, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.ConvTranspose(features=3, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
        x = nn.tanh(x)
        return x

# Define the discriminator network
class Discriminator(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Conv(features=128, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Conv(features=256, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Conv(features=1, kernel_size=(4, 4), strides=(1, 1), padding="VALID")(x)
        return x

# Define the loss functions
def generator_loss(params_generator, params_discriminator, images, masks):
    masked_images = images * masks
    generated_images = Generator().apply(params_generator, masked_images)
    discriminator_output = Discriminator().apply(params_discriminator, generated_images)
    loss = jnp.mean(jnp.square(discriminator_output - 1))
    return loss

def discriminator_loss(params_generator, params_discriminator, images, masks):
    masked_images = images * masks
    generated_images = Generator().apply(params_generator, masked_images)
    real_output = Discriminator().apply(params_discriminator, images)
    fake_output = Discriminator().apply(params_discriminator, generated_images)
    real_loss = jnp.mean(jnp.square(real_output - 1))
    fake_loss = jnp.mean(jnp.square(fake_output))
    total_loss = (real_loss + fake_loss) / 2
    return total_loss

# Define the train step
@jax.jit
def train_step(state_generator, state_discriminator, images, masks):
    # Update generator
    loss_generator, grads_generator = jax.value_and_grad(generator_loss)(
        state_generator.params, state_discriminator.params, images, masks)
    state_generator = state_generator.apply_gradients(grads=grads_generator)

    # Update discriminator
    loss_discriminator, grads_discriminator = jax.value_and_grad(discriminator_loss)(
        state_generator.params, state_discriminator.params, images, masks)
    state_discriminator = state_discriminator.apply_gradients(grads=grads_discriminator)

    return state_generator, state_discriminator, loss_generator, loss_discriminator

# Load and preprocess the dataset
def load_dataset(dataset_name, batch_size):
    ds = load_dataset(dataset_name, split="train")
    ds = ds.map(lambda x: (x["image"] / 127.5 - 1))
    ds = ds.cache().shuffle(1000).batch(batch_size).prefetch(1)
    return ds

# Create random masks
def create_random_masks(images, mask_size):
    batch_size = images.shape[0]
    mask_height, mask_width = mask_size
    masks = jnp.ones_like(images)
    
    for i in range(batch_size):
        y = np.random.randint(0, images.shape[1] - mask_height)
        x = np.random.randint(0, images.shape[2] - mask_width)
        masks = masks.at[i, y:y+mask_height, x:x+mask_width, :].set(0)
    
    return masks

# Set hyperparameters
num_epochs = 100
batch_size = 64
learning_rate = 2e-4
mask_size = (32, 32)

# Load the dataset
dataset = load_dataset("celeb_a", batch_size)

# Initialize the generator and discriminator
generator = Generator()
discriminator = Discriminator()

generator_params = generator.init(jax.random.PRNGKey(0), jnp.zeros((1, 128, 128, 3)))
discriminator_params = discriminator.init(jax.random.PRNGKey(1), jnp.zeros((1, 128, 128, 3)))

# Create train state
tx = optax.adam(learning_rate)
state_generator = train_state.TrainState.create(apply_fn=generator.apply, params=generator_params, tx=tx)
state_discriminator = train_state.TrainState.create(apply_fn=discriminator.apply, params=discriminator_params, tx=tx)

# Training loop
for epoch in range(num_epochs):
    for batch in dataset:
        masks = create_random_masks(batch, mask_size)
        state_generator, state_discriminator, loss_generator, loss_discriminator = train_step(
            state_generator, state_discriminator, batch, masks)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Generator Loss: {loss_generator:.4f}, "
          f"Discriminator Loss: {loss_discriminator:.4f}")

# Inpainting example
image = next(iter(dataset))
mask = create_random_masks(image, mask_size)
masked_image = image * mask

generated_image = Generator().apply(state_generator.params, masked_image)

plt.subplot(1, 3, 1)
plt.imshow(masked_image[0])
plt.title("Masked Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(generated_image[0])
plt.title("Generated Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(image[0])
plt.title("Original Image")
plt.axis("off")

plt.tight_layout()
plt.show()
