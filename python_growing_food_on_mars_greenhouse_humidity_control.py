# Python Growing Food on Mars: Monitoring and Adjusting Humidity Levels in a Greenhouse

# Welcome to the fourth class in the "Growing Food on Mars" series!
# In this class, we will learn how to monitor and adjust humidity levels in our Martian greenhouse using Python.

# First, let's import the necessary modules:
import random
import time

# Let's define a function to simulate the humidity level inside the greenhouse:
def greenhouse_humidity():
   # For simplicity, we'll assume the humidity level fluctuates between 30% and 70%
   return random.randint(30, 70)

# Now, let's create a function to monitor and adjust the humidity level:
def humidity_control(target_humidity):
   while True:
       current_humidity = greenhouse_humidity()
       print(f"Current humidity: {current_humidity}%")

       if current_humidity > target_humidity:
           print("Humidity is too high. Activating dehumidification system.")
           # Simulate dehumidifying the greenhouse
           time.sleep(2)
       elif current_humidity < target_humidity:
           print("Humidity is too low. Activating humidification system.")
           # Simulate humidifying the greenhouse
           time.sleep(2)
       else:
           print("Humidity is within the optimal range.")

       # Wait for a short time before checking the humidity again
       time.sleep(5)

# Let's set our target humidity level and start the humidity control system:
target_humidity_level = 50  # percent
print(f"Starting humidity control with a target humidity level of {target_humidity_level}%.")
humidity_control(target_humidity_level)

# Follow along and rewrite this code in your own Python file.
# Feel free to experiment with different target humidity levels and observe how the system responds.

# In the next class, we will explore how to optimize artificial lighting for plant growth on Mars using Python.
# Get ready to dive deeper into creating the perfect growing conditions for our crops on the Red Planet!
