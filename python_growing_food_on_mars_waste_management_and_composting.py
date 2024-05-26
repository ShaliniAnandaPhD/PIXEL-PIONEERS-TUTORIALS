# Python Growing Food on Mars: Waste Management and Composting System

# Welcome to the seventeenth class in the "Growing Food on Mars" series!
# In this class, we will learn how to create a Python-based system for waste management and composting on Mars.

# Throughout the previous classes, we have explored various Python concepts and tools, such as:
# - Variables and data types
# - Functions and parameters
# - Loops (for and while)
# - Conditional statements (if, elif, else)
# - Lists and dictionaries
# - Classes and objects
# - Modules (random and time)

# In this class, we will introduce a new Python syntax: Enumeration (enum)

# First, let's import the necessary modules:
import random
import time
from enum import Enum

# Let's create an enumeration for the type of waste:
class WasteType(Enum):
   ORGANIC = 1
   INORGANIC = 2

# Let's create a class to represent waste:
class Waste:
   def __init__(self, waste_type):
       self.waste_type = waste_type

# Let's create a class to represent the composter:
class Composter:
   def __init__(self, capacity):
       self.capacity = capacity
       self.waste = []

   def add_waste(self, waste):
       if len(self.waste) < self.capacity:
           self.waste.append(waste)
           print(f"Added {waste.waste_type.name} waste to the composter.")
       else:
           print("Composter is full. Cannot add more waste.")

   def compost(self):
       print("Composting waste...")
       time.sleep(3)  # Simulate time taken for composting
       self.waste = []  # Empty the composter after composting
       print("Composting complete.")

# Let's create a function to generate random waste:
def generate_waste():
   waste_type = random.choice(list(WasteType))
   return Waste(waste_type)

# Let's create an instance of the composter:
composter = Composter(5)  # Composter with a capacity of 5 units

# Let's simulate waste management and composting for 7 days:
num_days = 7
for day in range(1, num_days + 1):
   print(f"\nDay {day}:")
   num_waste = random.randint(1, 3)  # Generate 1-3 units of waste per day
   for _ in range(num_waste):
       waste = generate_waste()
       composter.add_waste(waste)
   
   if day % 3 == 0:  # Compost every 3 days
       composter.compost()

# In this class, we introduced the enum module, which allows us to define a set of named constants.
# We used Enum to create the WasteType enumeration, representing the different types of waste (ORGANIC and INORGANIC).
# Enumerations provide a way to assign meaningful names to a group of related constants, making the code more readable and maintainable.

# Follow along and rewrite this code in your own Python file.
# Feel free to experiment with different composter capacities, waste generation rates, and composting frequencies to further explore the waste management and composting system on Mars.

# In the next class, we will explore how to use Python to analyze and optimize energy consumption in a Martian greenhouse.
# Get ready to dive deeper into the world of sustainable agriculture on Mars!
