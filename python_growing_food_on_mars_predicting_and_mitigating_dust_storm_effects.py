# Python Growing Food on Mars: Predicting and Mitigating the Effects of Martian Dust Storms on Greenhouse Operations

# Welcome to the nineteenth class in the "Growing Food on Mars" series!
# In this class, we will learn how to develop a Python script for predicting and mitigating the effects of Martian dust storms on greenhouse operations.

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

# In this class, we will introduce a new Python syntax: Lambda Functions

# First, let's import the necessary modules:
import random

# Let's create a class to represent a dust storm:
class DustStorm:
   def __init__(self, severity):
       self.severity = severity

   def __str__(self):
       return f"Dust Storm (Severity: {self.severity})"

# Let's create a function to generate a random dust storm:
def generate_dust_storm():
   severity = random.randint(1, 10)
   return DustStorm(severity)

# Let's create a class to represent the greenhouse:
class Greenhouse:
   def __init__(self, structural_integrity):
       self.structural_integrity = structural_integrity

   def mitigate_damage(self, dust_storm):
       damage = dust_storm.severity * 0.1
       self.structural_integrity -= damage
       print(f"Greenhouse structural integrity decreased by {damage} due to the dust storm.")

   def reinforce_structure(self):
       reinforcement = 2
       self.structural_integrity += reinforcement
       print(f"Greenhouse structural integrity increased by {reinforcement} after reinforcement.")

# Let's create a function to predict the impact of dust storms using lambda:
predict_impact = lambda storm, greenhouse: f"Predicted impact: {storm.severity * 0.1:.2f} on greenhouse with structural integrity {greenhouse.structural_integrity}"

# Let's create an instance of the greenhouse:
greenhouse = Greenhouse(100)

# Let's simulate a series of dust storms and mitigate their effects:
num_storms = 5
for _ in range(num_storms):
   dust_storm = generate_dust_storm()
   print(dust_storm)
   print(predict_impact(dust_storm, greenhouse))
   greenhouse.mitigate_damage(dust_storm)
   print(f"Greenhouse structural integrity: {greenhouse.structural_integrity:.2f}")
   
   if greenhouse.structural_integrity < 50:
       greenhouse.reinforce_structure()
   
   print()

# In this class, we introduced the concept of lambda functions, which are small, anonymous functions that can be defined in a single line.
# We used a lambda function to create the predict_impact function, which takes a dust storm and a greenhouse as parameters and returns a string predicting the impact of the storm on the greenhouse.
# Lambda functions are useful for creating simple, one-time functions without the need for a formal function definition using the def keyword.

# Follow along and rewrite this code in your own Python file.
# Feel free to experiment with different dust storm severities, greenhouse structural integrities, and mitigation strategies to further explore the prediction and mitigation of the effects of Martian dust storms on greenhouse operations.

# In the next class, we will explore how to create a Python-based dashboard for real-time monitoring and control of a Martian greenhouse.
# Get ready to dive deeper into the world of advanced greenhouse management on Mars!
