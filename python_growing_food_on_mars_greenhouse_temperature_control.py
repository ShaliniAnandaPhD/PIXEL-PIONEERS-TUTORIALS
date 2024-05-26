# Python Growing Food on Mars: Automating Temperature Control in a Greenhouse

# Welcome to the third class in the "Growing Food on Mars" series!
# In this class, we will learn how to automate temperature control in our Martian greenhouse using Python.

# First, let's import the necessary modules:
import random
import time

# Let's define a function to simulate the temperature inside the greenhouse:
def greenhouse_temperature():
   # For simplicity, we'll assume the temperature fluctuates between 15 and 30 degrees Celsius
   return random.randint(15, 30)

# Now, let's create a function to control the temperature:
def temperature_control(target_temp):
   while True:
       current_temp = greenhouse_temperature()
       print(f"Current temperature: {current_temp}°C")

       if current_temp > target_temp:
           print("Temperature is too high. Activating cooling system.")
           # Simulate cooling the greenhouse
           time.sleep(2)
       elif current_temp < target_temp:
           print("Temperature is too low. Activating heating system.")
           # Simulate heating the greenhouse
           time.sleep(2)
       else:
           print("Temperature is within the optimal range.")

       # Wait for a short time before checking the temperature again
       time.sleep(5)

# Let's set our target temperature and start the temperature control system:
target_temperature = 25  # degrees Celsius
print(f"Starting temperature control with a target temperature of {target_temperature}°C.")
temperature_control(target_temperature)

# Follow along and rewrite this code in your own Python file.
# Feel free to experiment with different target temperatures and observe how the system responds.

# In the next class, we will explore how to monitor and adjust humidity levels in our Martian greenhouse using Python.
# Get ready to dive deeper into creating an optimal growing environment for our crops on Mars!
