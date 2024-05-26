# Python Growing Food on Mars: Monitoring and Maintaining Atmospheric Pressure in a Martian Greenhouse

# Welcome to the sixteenth class in the "Growing Food on Mars" series!
# In this class, we will learn how to develop a Python script for monitoring and maintaining atmospheric pressure in a Martian greenhouse.

# First, let's import the necessary modules:
import random
import time

# Let's create a function to simulate the atmospheric pressure in the greenhouse:
def simulate_pressure():
   return random.uniform(0.5, 1.5)  # Simulate pressure between 0.5 and 1.5 atm

# Let's create a function to adjust the atmospheric pressure:
def adjust_pressure(current_pressure, target_pressure):
   if current_pressure < target_pressure:
       print("Increasing atmospheric pressure...")
       time.sleep(2)  # Simulate time taken to increase pressure
       current_pressure += 0.1  # Increase pressure by 0.1 atm
   elif current_pressure > target_pressure:
       print("Decreasing atmospheric pressure...")
       time.sleep(2)  # Simulate time taken to decrease pressure
       current_pressure -= 0.1  # Decrease pressure by 0.1 atm
   return current_pressure

# Let's create a function to monitor and maintain the atmospheric pressure:
def maintain_pressure(target_pressure, duration):
   print(f"Monitoring atmospheric pressure for {duration} hours.")
   for hour in range(1, duration + 1):
       current_pressure = simulate_pressure()
       print(f"Hour {hour}: Current pressure: {current_pressure:.2f} atm")
       if abs(current_pressure - target_pressure) >= 0.1:
           current_pressure = adjust_pressure(current_pressure, target_pressure)
           print(f"Adjusted pressure: {current_pressure:.2f} atm")
       time.sleep(1)  # Simulate time between each hour

# Let's set the target atmospheric pressure and monitoring duration:
target_pressure = 1.0  # Earth-like atmospheric pressure (1 atm)
duration = 24  # Monitor for 24 hours

# Let's start monitoring and maintaining the atmospheric pressure:
maintain_pressure(target_pressure, duration)

# Follow along and rewrite this code in your own Python file.
# Feel free to experiment with different target pressures and monitoring durations to further explore the atmospheric pressure control system in a Martian greenhouse.

# In the next class, we will explore how to create a Python-based system for waste management and composting on Mars.
# Get ready to dive deeper into the world of sustainable agriculture on the Red Planet!
