# File name: keras_food_classification_cnn_nutrition.py
# File library: Keras, TensorFlow, OpenCV
# Use case: Nutrition - Food Classification

import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define image size and number of classes
img_size = (224, 224)
num_classes = 5

# Simulate food image data
def simulate_food_data(num_samples_per_class):
    food_classes = ['pizza', 'burger', 'sushi', 'pasta', 'salad']
    
    images = []
    labels = []
    
    for class_idx, food_class in enumerate(food_classes):
        for _ in range(num_samples_per_class):
            image = np.random.rand(img_size[0], img_size[1], 3) * 255
            image = image.astype(np.uint8)
            images.append(image)
            labels.append(class_idx)
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

# Generate simulated food data
num_samples_per_class = 100
images, labels = simulate_food_data(num_samples_per_class)

# Preprocess the data
images = images / 255.0
labels = keras.utils.to_categorical(labels, num_classes)

# Split the data into training and testing sets
train_size = int(0.8 * len(images))
train_images, test_images = images[:train_size], images[train_size:]
train_labels, test_labels = labels[:train_size], labels[train_size:]

# Define the CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc:.4f}')

# Save the trained model
model.save('food_classification_model.h5')

# Load the trained model
loaded_model = keras.models.load_model('food_classification_model.h5')

# Classify new food images
new_images_dir = 'path/to/new/food/images/'
new_images = []

for filename in os.listdir(new_images_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(new_images_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, img_size)
        new_images.append(image)

new_images = np.array(new_images) / 255.0
predictions = loaded_model.predict(new_images)

food_classes = ['pizza', 'burger', 'sushi', 'pasta', 'salad']
for i, pred in enumerate(predictions):
    class_idx = np.argmax(pred)
    class_name = food_classes[class_idx]
    print(f'Image {i+1} is classified as: {class_name}')
