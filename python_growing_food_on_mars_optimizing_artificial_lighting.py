# Python Growing Food on Mars: Optimizing Artificial Lighting for Plant Growth

# Welcome to the fifth class in the "Growing Food on Mars" series!
# In this class, we will learn how to optimize artificial lighting for plant growth on Mars using Python.

# First, let's import the necessary modules:
import time

# Let's define a function to simulate the growth rate of plants based on the light intensity:
def plant_growth_rate(light_intensity):
   # For simplicity, we'll assume the growth rate is directly proportional to the light intensity
   return light_intensity * 0.05

# Now, let's create a function to control the artificial lighting:
def lighting_control(light_intensity, hours):
   print(f"Setting artificial light intensity to {light_intensity}% for {hours} hours.")
   for hour in range(hours):
       growth_rate = plant_growth_rate(light_intensity)
       print(f"Hour {hour + 1}: Plant growth rate is {growth_rate:.2f}%.")
       time.sleep(1)  # Simulate the passage of time

# Let's define a function to optimize the lighting schedule:
def optimize_lighting():
   # We'll assume that plants need 12 hours of light per day
   total_hours = 12

   # We'll split the lighting schedule into three 4-hour periods with different intensities
   morning_intensity = 80
   afternoon_intensity = 100
   evening_intensity = 60

   print("Starting optimized lighting schedule.")
   lighting_control(morning_intensity, total_hours // 3)
   lighting_control(afternoon_intensity, total_hours // 3)
   lighting_control(evening_intensity, total_hours // 3)

# Let's start the optimized lighting schedule:
optimize_lighting()

# Follow along and rewrite this code in your own Python file.
# Feel free to experiment with different lighting intensities and durations to see how they affect plant growth.

# In the next class, we will explore how to simulate Martian soil composition and its effects on plant growth using Python.
# Get ready to dive deeper into understanding the unique challenges of growing crops on the Red Planet!
