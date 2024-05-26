# Python Growing Food on Mars: Creating a System for Pollination in a Martian Greenhouse

# Welcome to the thirteenth class in the "Growing Food on Mars" series!
# In this class, we will learn how to create a Python-based system for pollination in a Martian greenhouse.

# First, let's create a class to represent a pollinator robot:
class PollinatorRobot:
   def __init__(self, name):
       self.name = name
       self.pollinated_plants = []

   def pollinate(self, plant):
       print(f"{self.name} is pollinating {plant.name}.")
       plant.pollinated = True
       self.pollinated_plants.append(plant)

# Let's create a class to represent a plant:
class Plant:
   def __init__(self, name, days_to_pollinate):
       self.name = name
       self.days_to_pollinate = days_to_pollinate
       self.pollinated = False

   def needs_pollination(self, day):
       return day >= self.days_to_pollinate and not self.pollinated

# Let's create a function to simulate a day in the greenhouse:
def greenhouse_day(day, plants, robots):
   print(f"\nDay {day} in the Martian Greenhouse:")
   for plant in plants:
       if plant.needs_pollination(day):
           available_robot = next((robot for robot in robots if len(robot.pollinated_plants) < 3), None)
           if available_robot:
               available_robot.pollinate(plant)
           else:
               print(f"No available robots to pollinate {plant.name}.")

# Let's create some sample plants and pollinator robots:
tomato = Plant("Tomato", 30)
lettuce = Plant("Lettuce", 25)
carrot = Plant("Carrot", 35)
cucumber = Plant("Cucumber", 28)

plants = [tomato, lettuce, carrot, cucumber]

robot1 = PollinatorRobot("PollinatorBot-1")
robot2 = PollinatorRobot("PollinatorBot-2")

robots = [robot1, robot2]

# Let's simulate 40 days in the Martian greenhouse:
for day in range(1, 41):
   greenhouse_day(day, plants, robots)

# Let's print the pollination summary:
print("\nPollination Summary:")
for robot in robots:
   print(f"{robot.name} pollinated: {', '.join(plant.name for plant in robot.pollinated_plants)}")

# Follow along and rewrite this code in your own Python file.
# Feel free to experiment with different plant types, pollination schedules, and the number of pollinator robots to further explore the pollination system in a Martian greenhouse.

# In the next class, we will explore how to use Python to optimize space utilization and crop rotation in a Martian greenhouse.
# Get ready to dive deeper into the world of efficient agriculture on Mars!
