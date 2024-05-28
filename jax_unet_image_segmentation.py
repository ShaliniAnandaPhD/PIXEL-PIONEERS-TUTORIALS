# jax_unet_image_segmentation.py

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow.keras.datasets import oxford_iiit_pet

# Define the U-Net model
def jax_unet(inputs, num_classes):
    # Encoder (Downsampling)
    conv1 = jax.lax.conv(inputs, kernel_size=(3, 3), feature_map_shape=(64,), padding='SAME')
    conv1 = jax.nn.relu(conv1)
    conv1 = jax.lax.conv(conv1, kernel_size=(3, 3), feature_map_shape=(64,), padding='SAME')
    conv1 = jax.nn.relu(conv1)
    pool1 = jax.lax.avg_pool(conv1, window_shape=(2, 2), strides=(2, 2), padding='VALID')

    conv2 = jax.lax.conv(pool1, kernel_size=(3, 3), feature_map_shape=(128,), padding='SAME')
    conv2 = jax.nn.relu(conv2)
    conv2 = jax.lax.conv(conv2, kernel_size=(3, 3), feature_map_shape=(128,), padding='SAME')
    conv2 = jax.nn.relu(conv2)
    pool2 = jax.lax.avg_pool(conv2, window_shape=(2, 2), strides=(2, 2), padding='VALID')

    # Bridge
    conv3 = jax.lax.conv(pool2, kernel_size=(3, 3), feature_map_shape=(256,), padding='SAME')
    conv3 = jax.nn.relu(conv3)
    conv3 = jax.lax.conv(conv3, kernel_size=(3, 3), feature_map_shape=(256,), padding='SAME')
    conv3 = jax.nn.relu(conv3)

    # Decoder (Upsampling)
    up4 = jax.image.resize(conv3, (conv3.shape[0], conv3.shape[1] * 2, conv3.shape[2] * 2, conv3.shape[3]), method='nearest')
    concat4 = jnp.concatenate([up4, conv2], axis=-1)
    conv4 = jax.lax.conv(concat4, kernel_size=(3, 3), feature_map_shape=(128,), padding='SAME')
    conv4 = jax.nn.relu(conv4)
    conv4 = jax.lax.conv(conv4, kernel_size=(3, 3), feature_map_shape=(128,), padding='SAME')
    conv4 = jax.nn.relu(conv4)

    up5 = jax.image.resize(conv4, (conv4.shape[0], conv4.shape[1] * 2, conv4.shape[2] * 2, conv4.shape[3]), method='nearest')
    concat5 = jnp.concatenate([up5, conv1], axis=-1)
    conv5 = jax.lax.conv(concat5, kernel_size=(3, 3), feature_map_shape=(64,), padding='SAME')
    conv5 = jax.nn.relu(conv5)
    conv5 = jax.lax.conv(conv5, kernel_size=(3, 3), feature_map_shape=(64,), padding='SAME')
    conv5 = jax.nn.relu(conv5)

    # Output
    outputs = jax.lax.conv(conv5, kernel_size=(1, 1), feature_map_shape=(num_classes,), padding='SAME')
    outputs = jax.nn.softmax(outputs, axis=-1)

    return outputs

# Define the loss function
def jax_loss_fn(params, inputs, targets):
    predictions = jax_unet(inputs, num_classes=3)
    loss = jnp.mean(jax.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=targets))
    return loss

# Define the accuracy metric
def jax_accuracy(params, inputs, targets):
    predictions = jax_unet(inputs, num_classes=3)
    predicted_labels = jnp.argmax(predictions, axis=-1)
    target_labels = jnp.argmax(targets, axis=-1)
    accuracy = jnp.mean(predicted_labels == target_labels)
    return accuracy

# Prepare the data
def prepare_data(images, masks, train_size):
    images = images / 255.0
    masks = jax.nn.one_hot(masks, num_classes=3)
    train_images, test_images = images[:train_size], images[train_size:]
    train_masks, test_masks = masks[:train_size], masks[train_size:]
    return (train_images, train_masks), (test_images, test_masks)

# Train the model
def jax_train(params, optimizer, train_data, num_epochs, batch_size):
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        for i in range(0, len(train_data[0]), batch_size):
            batch_images = train_data[0][i:i+batch_size]
            batch_masks = train_data[1][i:i+batch_size]
            loss_value, grads = jax.value_and_grad(jax_loss_fn)(params, batch_images, batch_masks)
            params = optimizer.update(grads, params)
            epoch_loss += loss_value
            epoch_accuracy += jax_accuracy(params, batch_images, batch_masks)
        epoch_loss /= (len(train_data[0]) // batch_size)
        epoch_accuracy /= (len(train_data[0]) // batch_size)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    return params

# Example usage
(train_images, train_masks), (test_images, test_masks) = oxford_iiit_pet.load_data()
train_data, test_data = prepare_data(train_images, train_masks, train_size=3000)

rng = jax.random.PRNGKey(0)
params = jax.random.normal(rng, (train_images.shape[1], train_images.shape[2], 3))

optimizer = jax.optim.Adam(learning_rate=0.001)

params = jax_train(params, optimizer, train_data, num_epochs=10, batch_size=32)

test_accuracy = jax_accuracy(params, test_data[0], test_data[1])
print("Test Accuracy:", test_accuracy)

# Possible Errors and Solutions:

# ValueError: operands could not be broadcast together with shapes (x, y) (a, b)
# Solution: Ensure that the shapes of the predictions and targets match exactly when calculating the loss.

# ImportError: No module named 'jax'
# Solution: Ensure JAX is installed using `pip install jax`.

# TypeError: only integer scalar arrays can be converted to a scalar index
# Solution: Ensure that array indexing and slicing operations are correctly implemented.

# RuntimeError: Invalid argument: Non-scalable parameters
# Solution: Ensure all operations in the model are scalable and support JAX's JIT compilation.
