import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet', include_top=True)

# Define the food classes
food_classes = ['apple', 'banana', 'bread', 'carrot', 'egg', 'hamburger', 'orange', 'pizza', 'rice', 'salad']

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = preprocess_input(img_to_array(resized_frame))
    input_data = np.expand_dims(normalized_frame, axis=0)
    
    # Perform food recognition
    predictions = model.predict(input_data)
    predicted_class = food_classes[np.argmax(predictions)]
    
    # Display the predicted food class on the frame
    cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Diet Monitoring', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Possible Errors and Solutions:

# 1. Error: "Failed to grab frame"
#    Solution: Ensure the camera is properly connected and accessible. If using a laptop, ensure the webcam is not being used by another application.

# 2. Error: "ValueError: could not broadcast input array from shape (224,224,3) into shape (224,224)"
#    Solution: Ensure the input frame is correctly resized to the expected dimensions (224x224) and has three color channels.

# 3. Error: "ModuleNotFoundError: No module named 'cv2'"
#    Solution: Ensure OpenCV is installed. Use `pip install opencv-python-headless` to install it.

# 4. Error: "ModuleNotFoundError: No module named 'tensorflow'"
#    Solution: Ensure TensorFlow and Keras are installed. Use `pip install tensorflow` to install it.

# 5. Error: "IndexError: list index out of range"
#    Solution: Ensure the `food_classes` list is correctly defined and matches the output classes of the model.

