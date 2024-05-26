# Python Growing Food on Mars: Implementing a Pest Control System for Martian Agriculture

# Welcome to the fifteenth class in the "Growing Food on Mars" series!
# In this class, we will learn how to implement a Python-based pest control system for Martian agriculture.

# First, let's create a class to represent a pest:
class Pest:
   def __init__(self, name, damage_rate):
       self.name = name
       self.damage_rate = damage_rate

   def infest(self, crop):
       crop.health -= self.damage_rate
       print(f"{self.name} infested {crop.name}, reducing its health by {self.damage_rate}.")

# Let's create a class to represent a crop:
class Crop:
   def __init__(self, name, health):
       self.name = name
       self.health = health

   def is_alive(self):
       return self.health > 0

# Let's create a function to simulate pest infestation on a list of crops:
def simulate_pest_infestation(crops, pests):
   for crop in crops:
       for pest in pests:
           if crop.is_alive():
               pest.infest(crop)
           else:
               print(f"{crop.name} has already been destroyed.")

# Let's create a function to apply pest control measures:
def apply_pest_control(crops, pests, threshold):
   for crop in crops:
       if crop.health <= threshold:
           for pest in pests:
               if pest in crop.pests:
                   crop.pests.remove(pest)
                   print(f"Pest control applied to {crop.name}, removing {pest.name}.")

# Let's create some sample crops and pests:
tomato = Crop("Tomato", 100)
lettuce = Crop("Lettuce", 100)
carrot = Crop("Carrot", 100)

aphid = Pest("Aphid", 10)
whitefly = Pest("Whitefly", 15)
thrip = Pest("Thrip", 20)

crops = [tomato, lettuce, carrot]
pests = [aphid, whitefly, thrip]

# Let's simulate pest infestation for 3 rounds:
num_rounds = 3
for round in range(1, num_rounds + 1):
   print(f"\nRound {round}:")
   simulate_pest_infestation(crops, pests)

# Let's set the health threshold for applying pest control:
threshold = 50

# Let's apply pest control measures:
print("\nApplying pest control measures:")
apply_pest_control(crops, pests, threshold)

# Let's print the final crop health:
print("\nFinal Crop Health:")
for crop in crops:
   print(f"{crop.name}: {crop.health}")

# Follow along and rewrite this code in your own Python file.
# Feel free to experiment with different pest types, damage rates, and pest control thresholds to further enhance the pest control system for Martian agriculture.

# In the next class, we will explore how to develop a Python script for monitoring and maintaining atmospheric pressure in a Martian greenhouse.
# Get ready to dive deeper into the world of creating optimal growing conditions on Mars!
