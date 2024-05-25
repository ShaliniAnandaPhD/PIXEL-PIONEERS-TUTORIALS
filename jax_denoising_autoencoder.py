# File name: jax_denoising_autoencoder.py

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import linen as nn
from flax.training import train_state
import optax
from datasets import load_dataset

# Define the autoencoder model
class Autoencoder(nn.Module):
    latent_dim: int
    
    @nn.compact
    def __call__(self, x):
        # Encoder
        x = nn.Conv(32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.latent_dim)(x)
        
        # Decoder
        x = nn.Dense(7 * 7 * 64)(x)
        x = x.reshape((x.shape[0], 7, 7, 64))
        x = nn.Conv(64, kernel_size=(3, 3), transpose=True, strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(32, kernel_size=(3, 3), transpose=True, strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(1, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.sigmoid(x)
        return x

# Define the loss function
def mse_loss(params, batch):
    images, noisy_images = batch
    reconstructed_images = model.apply(params, noisy_images)
    loss = jnp.mean(jnp.square(reconstructed_images - images))
    return loss

# Define the evaluation metric
def psnr(params, batch):
    images, noisy_images = batch
    reconstructed_images = model.apply(params, noisy_images)
    mse = jnp.mean(jnp.square(reconstructed_images - images))
    psnr_value = 20 * jnp.log10(1.0 / jnp.sqrt(mse))
    return psnr_value

# Define the training step
@jax.jit
def train_step(state, batch):
    loss_value, grads = jax.value_and_grad(mse_loss)(state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss_value

# Define the evaluation step
@jax.jit
def eval_step(state, batch):
    loss_value = mse_loss(state.params, batch)
    psnr_value = psnr(state.params, batch)
    return loss_value, psnr_value

# Load and preprocess the dataset
def preprocess_images(examples):
    images = np.stack(examples['image'])
    images = images.astype(np.float32) / 255.0
    noisy_images = images + np.random.normal(0, 0.1, images.shape)
    noisy_images = np.clip(noisy_images, 0, 1)
    return images, noisy_images

dataset = load_dataset('mnist')
train_dataset = dataset['train'].map(preprocess_images, batched=True, batch_size=32)
test_dataset = dataset['test'].map(preprocess_images, batched=True, batch_size=32)

# Set hyperparameters
latent_dim = 128
learning_rate = 0.001
num_epochs = 10

# Initialize the model and optimizer
model = Autoencoder(latent_dim)
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 28, 28, 1)))
tx = optax.adam(learning_rate)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Training loop
for epoch in range(num_epochs):
    for batch in train_dataset:
        state, loss_value = train_step(state, batch)
    
    # Evaluation
    eval_loss = 0
    eval_psnr = 0
    num_eval_batches = 0
    for batch in test_dataset:
        loss_value, psnr_value = eval_step(state, batch)
        eval_loss += loss_value
        eval_psnr += psnr_value
        num_eval_batches += 1
    eval_loss /= num_eval_batches
    eval_psnr /= num_eval_batches
    
    print(f"Epoch {epoch + 1}, Train Loss: {loss_value:.4f}, Eval Loss: {eval_loss:.4f}, Eval PSNR: {eval_psnr:.2f}")

# Denoising example
noisy_image = next(iter(test_dataset))[1][0]
denoised_image = model.apply(state.params, noisy_image[jnp.newaxis, ...])[0]

# Visualize the results
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(noisy_image.squeeze(), cmap='gray')
axes[0].set_title('Noisy Image')
axes[0].axis('off')
axes[1].imshow(denoised_image.squeeze(), cmap='gray')
axes[1].set_title('Denoised Image')
axes[1].axis('off')
plt.tight_layout()
plt.show()
