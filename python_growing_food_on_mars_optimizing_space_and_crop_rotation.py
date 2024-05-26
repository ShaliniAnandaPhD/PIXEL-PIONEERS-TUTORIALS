# Python Growing Food on Mars: Optimizing Space Utilization and Crop Rotation

# Welcome to the fourteenth class in the "Growing Food on Mars" series!
# In this class, we will learn how to use Python to optimize space utilization and crop rotation in a Martian greenhouse.

# First, let's create a class to represent a crop:
class Crop:
   def __init__(self, name, days_to_harvest, space_required):
       self.name = name
       self.days_to_harvest = days_to_harvest
       self.space_required = space_required

# Let's create a function to calculate the total space required for a list of crops:
def calculate_total_space(crops):
   total_space = sum(crop.space_required for crop in crops)
   return total_space

# Let's create a function to find the crop with the shortest days to harvest:
def find_shortest_harvest_time(crops):
   return min(crops, key=lambda crop: crop.days_to_harvest)

# Let's create a function to generate a planting schedule based on space and harvest time:
def generate_planting_schedule(crops, available_space, num_cycles):
   planting_schedule = []
   for cycle in range(num_cycles):
       remaining_space = available_space
       cycle_crops = []
       while remaining_space > 0:
           shortest_harvest_crop = find_shortest_harvest_time(crops)
           if shortest_harvest_crop.space_required <= remaining_space:
               cycle_crops.append(shortest_harvest_crop)
               remaining_space -= shortest_harvest_crop.space_required
           else:
               break
       planting_schedule.append(cycle_crops)
       crops = [crop for crop in crops if crop not in cycle_crops]
   return planting_schedule

# Let's create some sample crops:
tomato = Crop("Tomato", 80, 2)
lettuce = Crop("Lettuce", 50, 1)
carrot = Crop("Carrot", 70, 1.5)
potato = Crop("Potato", 100, 3)

crops = [tomato, lettuce, carrot, potato]

# Let's set the available space in the Martian greenhouse:
available_space = 10  # square meters

# Let's generate a planting schedule for 3 cycles:
num_cycles = 3
planting_schedule = generate_planting_schedule(crops, available_space, num_cycles)

# Let's print the planting schedule:
print("Planting Schedule:")
for cycle, cycle_crops in enumerate(planting_schedule, start=1):
   print(f"Cycle {cycle}: {', '.join(crop.name for crop in cycle_crops)}")
   print(f"Space used: {calculate_total_space(cycle_crops)} square meters")
   print()

# Follow along and rewrite this code in your own Python file.
# Feel free to experiment with different crop types, available space, and the number of cycles to further optimize space utilization and crop rotation in a Martian greenhouse.

# In the next class, we will explore how to implement a Python-based pest control system for Martian agriculture.
# Get ready to dive deeper into the world of protecting our crops on Mars! 
