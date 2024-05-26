# File name: computer_vision_diet_monitoring_nutrition.py
# File library: OpenCV, TensorFlow, Keras
# Use case: Nutrition - Diet Monitoring

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
