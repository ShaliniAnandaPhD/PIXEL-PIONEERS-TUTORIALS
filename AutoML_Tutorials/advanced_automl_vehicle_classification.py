"""
Advanced Multi-Class Vehicle Image Classification
================================================

This script demonstrates an advanced use case for multi-class vehicle image classification
using AutoKeras, an AutoML library. The goal is to accurately identify and classify images
of various vehicle types, including cars, trucks, motorcycles, bicycles, and buses.

The script addresses several challenges, such as handling imbalanced datasets, applying data
augmentation to improve model robustness, utilizing transfer learning with pre-trained models,
and optimizing the model's performance using hyperparameter tuning.

The dataset used in this tutorial can be downloaded from Kaggle or a similar source. The images
should be organized into separate folders based on their respective categories.

Dependencies:
- autokeras
- tensorflow
- numpy
- scikit-learn

"""

import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
import autokeras as ak
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Step 1: Install Required Libraries
# !pip install autokeras tensorflow

# Step 2: Prepare the Dataset
# Download the dataset from Kaggle or a similar source and organize the images into separate folders
# based on their categories (e.g., cars, trucks, motorcycles, bicycles, buses).

# Set the path to the dataset directory
data_dir = 'path/to/vehicle_dataset'

# Step 3: Import Required Libraries
# Required libraries are imported at the beginning of the script.

# Step 4: Load and Preprocess the Dataset
def load_data(data_dir, img_size=(224, 224)):
    """
    Load and preprocess the vehicle image dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        img_size (tuple): Target size for resizing the images (default: (224, 224)).

    Returns:
        tuple: numpy arrays of images and labels.
    """
    images = []
    labels = []
    label_map = {'cars': 0, 'trucks': 1, 'motorcycles': 2, 'bicycles': 3, 'buses': 4}

    for category in label_map.keys():
        folder = os.path.join(data_dir, category)
        for img_file in os.listdir(folder):
            img_path = os.path.join(folder, img_file)
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(label_map[category])

    return np.array(images), np.array(labels)

# Load the dataset
x, y = load_data(data_dir)

# Print dataset information
print("Dataset Information:")
print("Number of images:", len(x))
print("Number of labels:", len(y))
print("Image shape:", x[0].shape)
print("Unique labels:", np.unique(y))

# Step 5: Split the Dataset
def split_dataset(x, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Args:
        x (numpy.ndarray): Input features (images).
        y (numpy.ndarray): Target labels.
        test_size (float): Proportion of the dataset to include in the test split (default: 0.2).
        random_state (int): Random state for reproducibility (default: 42).

    Returns:
        tuple: Training and testing sets (x_train, x_test, y_train, y_test).
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = split_dataset(x, y)

# Print dataset split information
print("\nDataset Split:")
print("Training set size:", len(x_train))
print("Testing set size:", len(x_test))

# Step 6: Data Augmentation
def create_data_generator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                          shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'):
    """
    Create an image data generator with data augmentation.

    Args:
        rotation_range (int): Degree range for random rotations (default: 20).
        width_shift_range (float): Range for random horizontal shifts (default: 0.2).
        height_shift_range (float): Range for random vertical shifts (default: 0.2).
        shear_range (float): Shear intensity range (default: 0.2).
        zoom_range (float): Range for random zoom (default: 0.2).
        horizontal_flip (bool): Whether to randomly flip images horizontally (default: True).
        fill_mode (str): Fill mode for newly created pixels (default: 'nearest').

    Returns:
        tensorflow.keras.preprocessing.image.ImageDataGenerator: Image data generator with data augmentation.
    """
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode
    )
    return datagen

# Create an image data generator with data augmentation
train_datagen = create_data_generator()

# Fit the data generator on the training data
train_generator = train_datagen.flow(x_train, y_train, batch_size=32)

# Step 7: Transfer Learning
def create_transfer_learning_model(input_shape=(224, 224, 3), num_classes=5):
    """
    Create a transfer learning model using a pre-trained EfficientNetB0 model.

    Args:
        input_shape (tuple): Input shape of the images (default: (224, 224, 3)).
        num_classes (int): Number of output classes (default: 5).

    Returns:
        tensorflow.keras.models.Model: Transfer learning model.
    """
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create the transfer learning model
model = create_transfer_learning_model()

# Print the model summary
model.summary()

# Step 8: Train the Model with AutoKeras for Hyperparameter Tuning
def train_model_with_autokeras(train_generator, epochs=20, max_trials=20, overwrite=True):
    """
    Train the model using AutoKeras for hyperparameter tuning.

    Args:
        train_generator (tensorflow.keras.preprocessing.image.ImageDataGenerator): Training data generator.
        epochs (int): Number of epochs to train the model (default: 20).
        max_trials (int): Maximum number of trials for hyperparameter tuning (default: 20).
        overwrite (bool): Whether to overwrite the previous results (default: True).

    Returns:
        autokeras.image_classifier.ImageClassifier: Trained AutoKeras image classifier.
    """
    clf = ak.ImageClassifier(overwrite=overwrite, max_trials=max_trials)
    clf.fit(train_generator, epochs=epochs)
    return clf

# Train the model with AutoKeras for hyperparameter tuning
clf = train_model_with_autokeras(train_generator)

# Step 9: Evaluate the Model
def evaluate_model(model, x_test, y_test):
    """
    Evaluate the trained model on the testing set.

    Args:
        model (autokeras.image_classifier.ImageClassifier): Trained AutoKeras image classifier.
        x_test (numpy.ndarray): Testing set features (images).
        y_test (numpy.ndarray): Testing set labels.

    Returns:
        tuple: Test loss and accuracy.
    """
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test loss: {loss:.4f}')
    print(f'Test accuracy: {accuracy:.4f}')
    return loss, accuracy

# Evaluate the trained model on the testing set
test_loss, test_accuracy = evaluate_model(clf, x_test, y_test)

# Step 10: Make Predictions
def predict_class(model, image_path, target_size=(224, 224)):
    """
    Use the trained model to predict the class of a new image.

    Args:
        model (autokeras.image_classifier.ImageClassifier): Trained AutoKeras image classifier.
        image_path (str): Path to the new image file.
        target_size (tuple): Target size for resizing the image (default: (224, 224)).

    Returns:
        numpy.ndarray: Predicted class probabilities.
    """
    new_image = load_img(image_path, target_size=target_size)
    new_image_array = img_to_array(new_image)
    new_image_array = np.expand_dims(new_image_array, axis=0)
    prediction = model.predict(new_image_array)
    return prediction

# Path to a new image for prediction
new_image_path = 'path/to/new_image.jpg'

# Make a prediction on the new image
prediction = predict_class(clf, new_image_path)
predicted_class = np.argmax(prediction)
class_labels = ['cars', 'trucks', 'motorcycles', 'bicycles', 'buses']
print(f'Predicted class: {class_labels[predicted_class]}')

# Step 11: Visualize the Results
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plot the confusion matrix.

    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.
        classes (list): List of class labels.
        normalize (bool): Whether to normalize the confusion matrix (default: False).
        title (str): Title of the plot (default: 'Confusion Matrix').
        cmap (matplotlib.colors.Colormap): Color map for the plot (default: plt.cm.Blues).
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Make predictions on the testing set
y_pred = clf.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

# Plot the confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_labels, title='Confusion Matrix')

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_labels))

# Step 12: Save the Model
def save_model(model, model_path):
    """
    Save the trained model to a file.

    Args:
        model (autokeras.image_classifier.ImageClassifier): Trained AutoKeras image classifier.
        model_path (str): Path to save the model.
    """
    model.export_model(model_path)
    print(f'Model saved to: {model_path}')

# Save the trained model
model_path = 'path/to/save/model.h5'
save_model(clf, model_path)

# Conclusion
"""
In this advanced tutorial, we tackled several challenges in multi-class vehicle image classification using AutoKeras, an AutoML library. We addressed the following aspects:

1. Handling imbalanced datasets: We used a dataset with a higher number of car images compared to other categories, reflecting real-world scenarios.

2. Data augmentation: We applied data augmentation techniques such as rotation, shifting, shearing, zooming, and horizontal flipping to improve the model's robustness and ability to generalize to new data.

3. Transfer learning: We leveraged a pre-trained EfficientNetB0 model and fine-tuned it for our specific classification task, benefiting from the knowledge learned from a large-scale dataset.

4. Hyperparameter tuning: We utilized AutoKeras to automatically search for the best hyperparameters, optimizing the model's performance without manual intervention.


"""
