# Python Growing Food on Mars: Creating a Dashboard for Real-Time Monitoring and Control of a Martian Greenhouse

# Welcome to the twentieth and final class in the "Growing Food on Mars" series!
# In this class, we will learn how to create a Python-based dashboard for real-time monitoring and control of a Martian greenhouse.

# Throughout the previous classes, we have explored various Python concepts and tools, such as:
# - Variables and data types
# - Functions and parameters
# - Loops (for and while)
# - Conditional statements (if, elif, else)
# - Lists and dictionaries
# - Classes and objects
# - Modules (random and time)
# - Enumeration (enum)
# - List Comprehension
# - Lambda Functions

# In this class, we will introduce a new Python library: Matplotlib

# First, let's import the necessary modules:
import random
import matplotlib.pyplot as plt

# Let's create a class to represent a sensor:
class Sensor:
   def __init__(self, name):
       self.name = name

   def read_value(self):
       return random.randint(0, 100)

# Let's create a class to represent the greenhouse dashboard:
class GreenhouseDashboard:
   def __init__(self, sensors):
       self.sensors = sensors

   def display_readings(self):
       readings = {sensor.name: sensor.read_value() for sensor in self.sensors}
       print("Sensor Readings:")
       for name, value in readings.items():
           print(f"{name}: {value}")
       return readings

   def plot_readings(self, readings):
       names = list(readings.keys())
       values = list(readings.values())

       plt.figure(figsize=(8, 6))
       plt.bar(names, values)
       plt.xlabel("Sensor")
       plt.ylabel("Value")
       plt.title("Greenhouse Sensor Readings")
       plt.show()

# Let's create some sensors:
temperature_sensor = Sensor("Temperature")
humidity_sensor = Sensor("Humidity")
co2_sensor = Sensor("CO2")
light_sensor = Sensor("Light")

sensors = [temperature_sensor, humidity_sensor, co2_sensor, light_sensor]

# Let's create an instance of the greenhouse dashboard:
dashboard = GreenhouseDashboard(sensors)

# Let's display and plot the sensor readings:
readings = dashboard.display_readings()
dashboard.plot_readings(readings)

# In this class, we introduced the Matplotlib library, which is a popular plotting library for Python.
# We used Matplotlib to create a bar chart visualization of the greenhouse sensor readings.
# The plt.figure() function is used to create a new figure, and the figsize parameter sets the size of the figure.
# The plt.bar() function is used to create a bar chart, with the sensor names on the x-axis and the sensor values on the y-axis.
# The plt.xlabel(), plt.ylabel(), and plt.title() functions are used to add labels and a title to the chart.
# Finally, the plt.show() function is called to display the plot.

# Follow along and rewrite this code in your own Python file.
# Feel free to experiment with different sensors, dashboard layouts, and visualization techniques to further enhance the real-time monitoring and control of a Martian greenhouse.

# Congratulations on completing the "Growing Food on Mars" series! You have learned a wide range of Python concepts and tools that can be applied to various aspects of Martian agriculture.
# Keep exploring and expanding your Python skills, and happy farming on Mars!
