# File name: jax_image_captioning_cnn_rnn.py

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import train_state
import optax
from datasets import load_dataset
from transformers import AutoFeatureExtractor

# Define the CNN encoder
class CNNEncoder(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        x = nn.Conv(features=64, kernel_size=(3, 3))(inputs)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=256, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        return x

# Define the RNN decoder
class RNNDecoder(nn.Module):
    vocab_size: int
    embedding_dim: int
    hidden_dim: int
    
    @nn.compact
    def __call__(self, inputs, hidden_state):
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embedding_dim)(inputs)
        x, hidden_state = nn.GRU(self.hidden_dim)(x, hidden_state)
        x = nn.Dense(self.vocab_size)(x)
        return x, hidden_state

# Define the CNN-RNN model
class ImageCaptioningModel(nn.Module):
    vocab_size: int
    embedding_dim: int
    hidden_dim: int
    
    @nn.compact
    def __call__(self, image, caption):
        # Encode the image using CNN
        image_features = CNNEncoder()(image)
        
        # Decode the caption using RNN
        caption_input = caption[:, :-1]
        caption_output = caption[:, 1:]
        hidden_state = image_features
        decoded_caption, _ = RNNDecoder(self.vocab_size, self.embedding_dim, self.hidden_dim)(caption_input, hidden_state)
        
        return decoded_caption, caption_output

# Define the loss function
def cross_entropy_loss(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
    return loss

# Define the evaluation metric (BLEU score)
def bleu_score(references, hypotheses):
    # Placeholder implementation, replace with actual BLEU score calculation
    return np.random.rand()

# Define the training step
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits, labels = ImageCaptioningModel(vocab_size, embedding_dim, hidden_dim).apply(params, batch["image"], batch["caption"])
        loss = cross_entropy_loss(logits, labels)
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = {"loss": cross_entropy_loss(logits, batch["caption"][:, 1:])}
    return state, metrics

# Define the evaluation step
@jax.jit
def eval_step(params, batch):
    logits, _ = ImageCaptioningModel(vocab_size, embedding_dim, hidden_dim).apply(params, batch["image"], batch["caption"])
    return {"loss": cross_entropy_loss(logits, batch["caption"][:, 1:])}

# Define the caption generation function
@jax.jit
def generate_caption(params, image, max_length):
    caption = jnp.array([1])  # Start with the <start> token
    hidden_state = CNNEncoder().apply(params, image)
    
    for _ in range(max_length):
        logits, hidden_state = RNNDecoder(vocab_size, embedding_dim, hidden_dim).apply(params, caption[:, -1], hidden_state)
        next_token = jnp.argmax(logits, axis=-1)
        caption = jnp.concatenate((caption, next_token), axis=0)
        
        if next_token == 2:  # <end> token
            break
    
    return caption

# Load and preprocess the dataset
dataset = load_dataset("coco_captions")
feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

def preprocess_data(example):
    image = feature_extractor(example["image"], return_tensors="np")["pixel_values"][0]
    caption = example["caption"].split()
    caption = [vocab["<start>"]] + [vocab[token] for token in caption] + [vocab["<end>"]]
    return {"image": image, "caption": caption}

# Set hyperparameters
vocab_size = 10000
embedding_dim = 256
hidden_dim = 512
learning_rate = 0.001
num_epochs = 10
batch_size = 32

# Preprocess the dataset
train_dataset = dataset["train"].map(preprocess_data, remove_columns=["id", "image", "caption"])
train_dataset = train_dataset.shuffle(seed=42).batch(batch_size)

# Create the model
model = ImageCaptioningModel(vocab_size, embedding_dim, hidden_dim)
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 224, 224, 3)), jnp.ones((1, 10), dtype=jnp.int32))["params"]

# Create the optimizer and training state
tx = optax.adam(learning_rate)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Training loop
for epoch in range(num_epochs):
    for batch in train_dataset:
        state, metrics = train_step(state, batch)
    
    print(f"Epoch {epoch + 1}, Loss: {metrics['loss']:.4f}")

# Generate captions for sample images
sample_images = next(iter(train_dataset))["image"][:5]
for image in sample_images:
    caption = generate_caption(state.params, image[jnp.newaxis, ...], max_length=20)
    caption_text = " ".join([vocab_inv[token] for token in caption if token not in [0, 1, 2]])
    print("Generated Caption:", caption_text)

# Possible Errors and Solutions:
# 1. Missing Vocabulary: Ensure the vocabulary includes all tokens in the captions.
# 2. Dimension Mismatch: Verify the dimensions of inputs and outputs in the model.
# 3. Gradient Issues: Check for NaNs in gradients and use gradient clipping if necessary.
# 4. Training Instability: Adjust learning rate or batch size to stabilize training.
