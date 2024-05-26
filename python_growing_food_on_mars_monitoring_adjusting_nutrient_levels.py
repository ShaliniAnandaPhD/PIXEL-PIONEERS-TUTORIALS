# Python Growing Food on Mars: Monitoring and Adjusting Nutrient Levels in Hydroponic Systems

# Welcome to the eighth class in the "Growing Food on Mars" series!
# In this class, we will learn how to develop a Python script for monitoring and adjusting nutrient levels in hydroponic systems.

# First, let's define a dictionary containing the optimal nutrient levels for our plants:
optimal_nutrient_levels = {
   "nitrogen": 150,
   "phosphorus": 50,
   "potassium": 200,
   "calcium": 150,
   "magnesium": 50
}

# Let's define a function to check the current nutrient levels in the hydroponic system:
def check_nutrient_levels(current_levels):
   for nutrient, level in current_levels.items():
       if level < optimal_nutrient_levels[nutrient] * 0.9:
           print(f"Warning: {nutrient} level is low at {level} ppm.")
       elif level > optimal_nutrient_levels[nutrient] * 1.1:
           print(f"Warning: {nutrient} level is high at {level} ppm.")
       else:
           print(f"{nutrient} level is optimal at {level} ppm.")

# Now, let's define a function to adjust the nutrient levels:
def adjust_nutrient_levels(current_levels):
   for nutrient, level in current_levels.items():
       if level < optimal_nutrient_levels[nutrient] * 0.9:
           amount_to_add = optimal_nutrient_levels[nutrient] - level
           print(f"Adding {amount_to_add} ppm of {nutrient} to the hydroponic system.")
           current_levels[nutrient] += amount_to_add
       elif level > optimal_nutrient_levels[nutrient] * 1.1:
           amount_to_remove = level - optimal_nutrient_levels[nutrient]
           print(f"Removing {amount_to_remove} ppm of {nutrient} from the hydroponic system.")
           current_levels[nutrient] -= amount_to_remove

# Let's create a dictionary to store the current nutrient levels in the hydroponic system:
current_nutrient_levels = {
   "nitrogen": 120,
   "phosphorus": 60,
   "potassium": 180,
   "calcium": 200,
   "magnesium": 40
}

# Let's check the current nutrient levels:
print("Checking current nutrient levels:")
check_nutrient_levels(current_nutrient_levels)

# Now, let's adjust the nutrient levels:
print("\nAdjusting nutrient levels:")
adjust_nutrient_levels(current_nutrient_levels)

# Let's recheck the nutrient levels after adjustment:
print("\nRechecking nutrient levels after adjustment:")
check_nutrient_levels(current_nutrient_levels)

# Follow along and rewrite this code in your own Python file.
# Feel free to experiment with different initial nutrient levels and observe how the system adjusts them.

# In the next class, we will explore how to use Python to analyze plant growth data and optimize yield on Mars.
# Get ready to dive deeper into the world of data-driven agriculture in Martian greenhouses!
